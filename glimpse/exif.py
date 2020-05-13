"""
Read and write exchangeable image file format (exif) metadata.
"""
import copy
import datetime
import piexif

class Exif:
    """
    Container and parser of image metadata.

    Provides access to the Exchangeable image file format (Exif) metadata tags
    embedded in an image file using :doc:`piexif <piexif:index>`.

    Arguments:
        path (str): Path to image
        thumbnail (bool): Whether to retain the image thumbnail

    Attributes:
        tags (dict): Exif tags returned by :func:`piexif.load`.
            The tags are grouped by their Image File Directory (IFD):

                - 'Exif' (Exif SubIFD): Image generation
                - '0th' (IFD0): Main image
                - '1st' (IFD1): Thumbnail image
                - 'GPS' (GPS IFD): Position and trajectory
                - 'Interop' (Interoperability IFD): Compatibility

            The thumbnail image, if present, is stored as `bytes` in 'thumbnail'.

        size (tuple): Image size in pixels (nx, ny).
            Parsed from 'PixelXDimension' and 'PixelYDimension', if present,
            or read by GDAL.
        datetime (datetime.datetime): Capture date and time.
            Parsed from 'DateTimeOriginal' and 'SubSecTimeOriginal'.
        exposure (float): Exposure time in seconds.
            Parsed from 'ExposureTime'.
        aperture (float): Aperture size as the f-number
            (https://wikipedia.org/wiki/F-number).
            Parsed from 'FNumber'.
        iso (float): Film speed following the ISO system
            (https://wikipedia.org/wiki/Film_speed#ISO).
            Parsed from 'ISOSpeedRatings'.
        fmm (float): Focal length in millimeters.
            Parsed from 'FocalLength'.
        make (str): Camera make.
            Parsed from 'Make'.
        model (str): Camera model.
            Parsed from 'Model'.
    """
    def __init__(self, path, thumbnail=False):
        self.tags = piexif.load(path, key_is_name=True)
        if not thumbnail:
            self.tags.pop('thumbnail', None)
            self.tags.pop('1st', None)

    @property
    def size(self):
        width = self.parse_tag('PixelXDimension')
        height = self.parse_tag('PixelYDimension')
        if width and height:
            return width, height
        else:
            return None

    @property
    def datetime(self):
        datetime_str = self.parse_tag('DateTimeOriginal')
        if not datetime_str:
            return None
        subsec_str = self.parse_tag('SubSecTimeOriginal')
        if subsec_str:
            return datetime.datetime.strptime(
                datetime_str + '.' + subsec_str, '%Y:%m:%d %H:%M:%S.%f')
        else:
            return datetime.datetime.strptime(
                datetime_str, '%Y:%m:%d %H:%M:%S')

    @property
    def exposure(self):
        return self.parse_tag('ExposureTime')

    @property
    def aperture(self):
        return self.parse_tag('FNumber')

    @property
    def iso(self):
        return self.parse_tag('ISOSpeedRatings')

    @property
    def fmm(self):
        return self.parse_tag('FocalLength')

    @property
    def make(self):
        return self.parse_tag('Make', group='0th')

    @property
    def model(self):
        return self.parse_tag('Model', group='0th')

    def parse_tag(self, tag, group='Exif'):
        """
        Return the parsed value of a tag.

        The following strategies are applied:

            - if `bytes`, decode to `str`
            - if `tuple` of length 2 (rational), convert to `float`
            - if `int`, convert to `float`

        Arguments:
            tag (str): Tag name
            group (str): Group name ('Exif', '0th', '1st', 'GPS', or 'Interop')

        Returns:
            object: Parsed tag value, or `None` if missing
        """
        try:
            value = self.tags[group][tag]
        except KeyError:
            return None
        if isinstance(value, bytes):
            return value.decode()
        elif isinstance(value, tuple) and len(value) == 2:
            return value[0] / value[1]
        elif isinstance(value, int):
            return float(value)
        else:
            return value

    def dump(self):
        """
        Return tags as bytes.

        The encoding is performed by :func:`piexif.dump`.
        The result can be embedded into an image file, for example using
        :func:`piexif.insert` or :meth:`PIL.Image.Image.save`.

        Returns:
            bytes: :attr:`tags` encoded as a byte string
        """
        # Copy tags before modifying inplace
        tags = copy.deepcopy(self.tags)
        # Replace key names with codes
        for group in self.tags:
            if group == 'thumbnail':
                continue
            if group not in ('0th', '1st', 'Exif', 'GPS', 'Interop'):
                raise ValueError('Invalid group \'{0}\''.format(group))
            ifd_name = 'ImageIFD' if group in ('0th', '1st') else group + 'IFD'
            ifd = getattr(piexif, ifd_name)
            for tag in self.tags[group]:
                try:
                    code = getattr(ifd, tag)
                except AttributeError:
                    raise ValueError('Invalid tag \'{0}\' in group \'{1}\''
                        .format(tag, group))
                tags[group][code] = tags[group].pop(tag)
        # Encode to bytes
        return piexif.dump(tags)

    def insert(self, path):
        piexif.insert(self.dump(), path)
