classdef Image
  % IMAGE Photographic image data structure.
  %
  % Image Properties:
  % cam      - Camera object
  %
  % Image Properties (read-only):
  % info     - File information from imfinfo
  % file     - Path to image file
  % date_str - Capture date and time as a string ('yyyy-mm-dd HH:MM:SS.FFF')
  % date_num - Cature date and time as a serial date number
  % shutter  - Shutter speed in seconds
  % aperture - Lens aperture
  % iso      - Film speed
  % ev       - Exposure value
  % gps      - GPS metadata
  % size     - Size of original image [nx|ncols|width, ny|nrows|height]
  %
  % Image Properties (dependent):
  % scale    - Scaling between original and camera image size
  %
  % Image Methods:
  % Image - Construct a new Image object
  % read  - Read image data
  %
  % See also imfinfo, datestr, datenum

  properties
    cam
    svg = struct();
    gcp = struct('uv', [], 'xyz', []);
    gcl = struct('uv', [], 'xyz', []);
    fixedpolys = {};
    freepolys = {};
    anchor = 0;
    I = [];
  end

  properties (SetAccess = private)
    info
  end

  properties (Dependent, SetAccess = private)
    file
    date_str
    date_num
    shutter
    aperture
    iso
    ev
    gps
    size
  end

  properties (Dependent)
    scale
  end

  methods

    function images = Image(files, cam)
      % IMAGE Construct a new Image object.
      %
      %   img = Image(files, cam = Camera())
      %
      % Image size, sensor size, and focal length are loaded from the file
      % unless overloaded by cam.
      %
      % Inputs:
      %   file - Path to image file
      %   cam  - Camera object

      % Check inputs
      if nargin < 1
        return
      end
      if isempty(files)
        temp = Image();
        images = temp(false);
        return
      end
      if ~isa(files, 'cell')
        files = {files};
      end
      if nargin < 2
        cam = Camera();
      end
      if nargin > 1 && ~isa(cam, 'Camera')
        error('Not an object of class Camera.');
      end
      % Expand paths
      files = cellfun(@expand_path, files, 'uniform', false);
      files = [files{:}];
      % Preallocate array
      images(length(files)) = Image();
      for i = 1:length(files)
        images(i) = foreach(images(i), files{i}, cam);
      end
      % For each file
      function img = foreach(img, file, cam)
        % Metadata
        img.info = imfinfo(file);
        % Camera
        img.cam = cam;
        % Image size in pixels [nx, ny]
        if isempty(img.cam.imgsz)
          img.cam.imgsz = [img.info.Width, img.info.Height];
        else
          % Check that target size is compatible with image size
          Camera.getScaleFromSize([img.info.Width, img.info.Height], img.cam.imgsz);
        end
        % Sensor size [mm width, mm height]
        if isempty(img.cam.sensorsz) && all(isfield(img.info, {'Make', 'Model'}))
          img.cam.sensorsz = Camera.getSensorSize(img.info.Make, img.info.Model);
        end
        % Focal length in mm (if not already set in pixels)
        if ~isempty(img.cam.sensorsz) && isempty(img.cam.f)
          if isfield(img.info.DigitalCamera, 'FocalLength')
            img.cam.fmm = img.info.DigitalCamera.FocalLength;
          elseif isfield(img.info.DigitalCamera, 'FocalLengthIn35mmFilm')
            % FIXME: Convert to true focal length using sensor size?
            img.cam.fmm = img.info.DigitalCamera.FocalLengthIn35mmFilm;
            warning('True focal length not found, using 35mm equivalent.');
          end
        end
        % SVG
        svg_path = change_ext(file, '.svg');
        if exist(svg_path, 'file')
          img.svg = svg2struct(svg_path);
        end
      end
    end

    function value = get.file(img)
      value = img.info.Filename;
    end

    function value = get.date_str(img)
      if isfield(img.info.DigitalCamera, 'DateTimeOriginal')
        date_time = strsplit(deblank(img.info.DigitalCamera.DateTimeOriginal), ' ');
        date_time{1} = strrep(date_time{1}, ':', '-');
        value = [date_time{1} ' ' date_time{2}];
        if isfield(img.info.DigitalCamera, 'SubsecTimeOriginal')
          value = [value '.' deblank(strtrim(img.info.DigitalCamera.SubsecTimeOriginal))];
        elseif isfield(img.info.DigitalCamera, 'SubsecTime')
          value = [value '.' deblank(strtrim(img.info.DigitalCamera.SubsecTime))];
        end
      else
        value = [];
      end
    end

    function value = get.date_num(img)
      format = 'yyyy-mm-dd HH:MM:SS';
      if ~isempty(strfind(img.date_str, '.'))
        format = [format '.FFF'];
      end
      value = datenum(img.date_str, format);
    end

    function value = get.shutter(img)
      if isfield(img.info.DigitalCamera, 'ExposureTime')
        value = img.info.DigitalCamera.ExposureTime; % (in seconds)
      else
        value = [];
      end
    end

    function value = get.aperture(img)
      if isfield(img.info.DigitalCamera, 'FNumber')
        value = img.info.DigitalCamera.FNumber;
      else
        value = [];
      end
    end

    function value = get.iso(img)
      if isfield(img.info.DigitalCamera, 'ISOSpeedRatings')
        value = img.info.DigitalCamera.ISOSpeedRatings;
      else
        value = [];
      end
    end

    function value = get.ev(img)
      % https://en.wikipedia.org/wiki/Exposure_value
      value = log2(1000 * img.aperture^2 / (img.iso * img.shutter));
    end

    function value = get.gps(img)
      value = [];
      if isfield(img.info, 'GPSInfo')
        % GPS Date & Time (serial date)
        % NOTE: Missing GPS SubsecTime field.
        if isfield(img.info.GPSInfo, 'GPSDateStamp')
          GPSDateStamp = deblank(strtrim(img.info.GPSInfo.GPSDateStamp));
          HMS = img.info.GPSInfo.GPSTimeStamp;
          value.datestr = [GPSDateStamp ' ' sprintf('%02d', HMS(1)) ':' sprintf('%02d', HMS(2)) ':' sprintf('%02d', HMS(3))];
          value.datenum = datenum(value.datestr, 'yyyy:mm:dd HH:MM:SS');
        end
        % GPS Position (lng, lat)
        if isfield(img.info.GPSInfo, {'GPSLatitude','GPSLongitude'})
          % Latitude (decimal degrees)
          lat = img.info.GPSInfo.GPSLatitude;
          lat = lat(1) + lat(2) / 60 + lat(3) / 3600;
          if deblank(strtrim(img.info.GPSInfo.GPSLatitudeRef)) == 'S'
            lat = -lat;
          end
          % Longitude (decimal degrees)
          lng = img.info.GPSInfo.GPSLongitude;
          lng = lng(1) + lng(2) / 60 + lng(3) / 3600;
          if deblank(strtrim(img.info.GPSInfo.GPSLongitudeRef)) == 'W'
            lng = -lng;
          end
          value.lnglat = [lng lat];
        end
        % GPS Altitude
        % NOTE: Unknown datum.
        if isfield(img.info.GPSInfo, 'GPSAltitude')
          value.altitude = img.info.GPSInfo.GPSAltitude; % (meters)
        end
        % GPS Satellites
        if isfield(img.info.GPSInfo, 'GPSSatellites')
          value.satellites = str2double(deblank(strtrim(img.info.GPSInfo.GPSSatellites)));
        end
      end
    end

    function value = get.size(img)
      value = [img.info.Width, img.info.Height];
    end

    function value = get.scale(img)
      if isempty(img.cam.imgsz)
        value = [];
      else
        value = Camera.getScaleFromSize(img.size, img.cam.imgsz);
      end
    end

    function img = set.scale(img, value)
      old_scale = img.scale;
      if isempty(old_scale)
        img.cam.imgsz = value * img.size;
      else
        img.cam = img.cam.resize(img.size);
        if value ~= 1
          img.cam = img.cam.resize(value);
        end
      end
    end

    function I = read(img, scale)
      % READ Read image data.
      %
      %   I = img.read(scale = img.scale)
      %
      % Inputs:
      %   scale - Resize scale factor

      if nargin < 2 || isempty(scale)
        scale = img.scale;
      end
      if isempty(img.I)
        I = imread(img.file);
      else
        I = img.I;
      end
      if ~isempty(scale) && scale ~= 1
        I = imresize(I, scale);
      end
    end

    function plot(img, control, polys)
      if nargin < 2 || isempty(control)
        control = false;
      end
      if nargin < 3 || isempty(polys)
        polys = false;
      end
      imshow(img.read());
      hold on
      if control
        % Point errors
        if ~isempty(img.gcp.uv) && ~isempty(img.gcp.xyz)
          duv = img.cam.projerror(img.gcp.uv, img.gcp.xyz);
          plot(img.gcp.uv(:, 1), img.gcp.uv(:, 2), 'g*');
          quiver(img.gcp.uv(:, 1), img.gcp.uv(:, 2), duv(:, 1), duv(:, 2), 0, 'r');
        end
        % Line errors
        if ~isempty(img.gcl.uv) && ~isempty(img.gcl.xyz)
          plot(img.gcl.uv(:, 1), img.gcl.uv(:, 2), 'g.');
          duv = img.cam.projerror_lines(img.gcl.uv, img.gcl.xyz);
          quiver(img.gcl.uv(:, 1), img.gcl.uv(:, 2), duv(:, 1), duv(:, 2), 0, 'r');
          for i_line = 1:length(img.gcl.xyz)
            xyz = img.cam.clip_line_inview(img.gcl.xyz{i_line});
            for i_segment = 1:length(xyz)
              pluv = img.cam.project(xyz{i_segment});
              plot(pluv(:, 1), pluv(:, 2), 'y-');
            end
          end
        end
      end
      if polys
        for poly = img.fixedpolys
          plot(poly{1}(:, 1), poly{1}(:, 2), 'y:');
        end
        for poly = img.freepolys
          plot(poly{1}(:, 1), poly{1}(:, 2), 'y:');
        end
      end
      hold off
    end

    function [I0, dxyz] = project(img, cam, dxyz)
      if any(cam.imgsz ~= img.size)
        error('Original and target image sizes must be equal.')
      end
      % Precompute reference grid.
      [u, v] = meshgrid(0.5:(cam.imgsz(1) - 0.5), 0.5:(cam.imgsz(2) - 0.5));
      if nargin < 3 || isempty(dxyz)
        dxyz = cam.invproject([u(:), v(:)]);
      end
      % Project and interpolate at grid points.
      puv = img.cam.project(dxyz, true);
      I = double(img.read());
      I0 = uint8(nan(size(I)));
      for channel = 1:size(I, 3)
        temp = interp2(u, v, I(:, :, channel), puv(:, 1), puv(:, 2), '*cubic');
        I0(:, :, channel) = reshape(temp, flip(cam.imgsz));
      end
    end

  end % methods

  methods (Static)

  end % methods (Static)

  methods (Access = private)

  end % methods (Access = private)

end % classdef
