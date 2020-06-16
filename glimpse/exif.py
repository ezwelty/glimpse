"""Read and write exchangeable image file format (exif) metadata."""
import copy
import datetime
from typing import Optional, Tuple, Union

import piexif

SENSOR_SIZES = {
    # https://www.dpreview.com/reviews/nikond2x/2
    "NIKON CORPORATION NIKON D2X": (23.7, 15.7),
    # https://www.dpreview.com/reviews/nikond200/2
    "NIKON CORPORATION NIKON D200": (23.6, 15.8),
    # https://www.dpreview.com/reviews/nikond300s/2
    "NIKON CORPORATION NIKON D300S": (23.6, 15.8),
    # https://www.dpreview.com/reviews/nikoncp8700/2
    "NIKON E8700": (8.8, 6.6),
    # https://www.dpreview.com/reviews/canoneos20d/2
    "Canon Canon EOS 20D": (22.5, 15.0),
    # https://www.dpreview.com/reviews/canoneos40d/2
    "Canon Canon EOS 40D": (22.2, 14.8),
}


class Exif:
    """
    Container and parser of image metadata.

    Provides access to the Exchangeable image file format (Exif) metadata tags
    embedded in an image file using :doc:`piexif <piexif:index>`.

    Example:
        >>> exif = Exif('tests/AK10b_20141013_020336.JPG')
        >>> exif.imgsz
        (800, 536)
        >>> exif.fmm
        20.0
        >>> exif.aperture
        8.0
        >>> exif.exposure
        0.0125
        >>> exif.iso
        200
        >>> exif.datetime
        datetime.datetime(2014, 10, 13, 2, 3, 36, 280000)
        >>> exif.make
        'NIKON CORPORATION'
        >>> exif.model
        'NIKON D200'
        >>> exif.sensorsz
        (23.6, 15.8)

    Arguments:
        path (str): Path to JPEG, TIFF, or WebP image.
        thumbnail (bool): Whether to retain the image thumbnail.

    Attributes:
        tags (dict): Exif tags read from the image. The tags are grouped by Image File
            Directory (IFD):

                - `Exif` (Exif SubIFD): Image generation
                - `0th` (IFD0): Main image
                - `1st` (IFD1): Thumbnail image
                - `GPS` (GPS IFD): Position and trajectory
                - `Interop` (Interoperability IFD): Compatibility

            The thumbnail image, if present,
            is stored as :py:class:`bytes` in`thumbnail`.
        imgsz (Tuple[int, int]): Image size in pixels (nx, ny).
            Parsed from `PixelXDimension` and `PixelYDimension`.
        datetime (datetime.datetime): Capture date and time.
            Parsed from `DateTimeOriginal` and `SubSecTimeOriginal`.
        exposure (float): Exposure time in seconds. Parsed from `ExposureTime`.
        aperture (float): Aperture size as the f-number
            (https://wikipedia.org/wiki/F-number). Parsed from `FNumber`.
        iso (int): Film speed following the ISO system
            (https://wikipedia.org/wiki/Film_speed#ISO). Parsed from `ISOSpeedRatings`.
        fmm (float): Focal length in millimeters. Parsed from `FocalLength`.
        make (str): Camera make. Parsed from `Make`.
        model (str): Camera model. Parsed from `Model`.
        sensorsz (Tuple[float, float]): Sensor size in millimeters (nx, ny).
            Data is from Digital Photography Review (https://dpreview.com) reviews and
            their article https://dpreview.com/articles/8095816568/sensorsizes.
    """

    def __init__(self, path: str, thumbnail: bool = False) -> None:
        self.tags = piexif.load(path, key_is_name=True)
        if not thumbnail:
            self.tags.pop("thumbnail", None)
            self.tags.pop("1st", None)

    @property
    def imgsz(self) -> Optional[Tuple[int, int]]:
        """Image size in pixels (nx, ny)."""
        width = self.parse_tag("PixelXDimension")
        height = self.parse_tag("PixelYDimension")
        if width and height:
            return int(width), int(height)
        return None

    @property
    def datetime(self) -> Optional[datetime.datetime]:
        """Capture date and time."""
        ymdhms = self.parse_tag("DateTimeOriginal")
        if not ymdhms:
            return None
        ss = self.parse_tag("SubSecTimeOriginal")
        if not ss:
            return datetime.datetime.strptime(str(ymdhms), "%Y:%m:%d %H:%M:%S")
        return datetime.datetime.strptime(
            str(ymdhms) + "." + str(ss), "%Y:%m:%d %H:%M:%S.%f"
        )

    @property
    def exposure(self) -> Optional[float]:
        """Exposure time in seconds."""
        value = self.parse_tag("ExposureTime")
        return float(value) if value else None

    @property
    def aperture(self) -> Optional[float]:
        """Aperture size as the f-number."""
        value = self.parse_tag("FNumber")
        return float(value) if value else None

    @property
    def iso(self) -> Optional[int]:
        """Film speed following the ISO system."""
        value = self.parse_tag("ISOSpeedRatings")
        return int(value) if value else None

    @property
    def fmm(self) -> Optional[float]:
        """Focal length in millimeters."""
        value = self.parse_tag("FocalLength")
        return float(value) if value else None

    @property
    def make(self) -> Optional[str]:
        """Camera make."""
        value = self.parse_tag("Make", group="0th")
        return str(value) if value else None

    @property
    def model(self) -> Optional[str]:
        """Camera model."""
        value = self.parse_tag("Model", group="0th")
        return str(value) if value else None

    @property
    def sensorsz(self) -> Optional[Tuple[float, float]]:
        """Sensor size in millimeters (nx, ny)."""
        if self.make and self.model:
            make_model = self.make.strip() + " " + self.model.strip()
            return SENSOR_SIZES.get(make_model)
        return None

    def parse_tag(
        self, tag: str, group: str = "Exif"
    ) -> Optional[Union[int, str, float]]:
        """
        Return the parsed value of a tag.

        Arguments:
            tag: Tag name.
            group: Group name ('Exif', '0th', '1st', 'GPS', or 'Interop').

        Returns:
            Parsed tag value.
        """
        try:
            value = self.tags[group][tag]
        except KeyError:
            return None
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, tuple) and len(value) == 2:
            return value[0] / value[1]
        return value

    def dump(self) -> bytes:
        r"""
        Return :attr:`tags` as a byte string.

        Returns:
            Exif tags encoded as a byte string.

        Raises:
            ValueError: :attr:`tags` contains an invalid key.

        Example:
            >>> exif = Exif('tests/AK10b_20141013_020336.JPG')
            >>> exif.tags = {'Exif': {}}
            >>> exif.dump()
            b'Exif\x00\x00MM\x00*\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00'
            >>> exif.tags = {'Unknown': {}}
            >>> exif.dump()
            Traceback (most recent call last):
                ...
            ValueError: Invalid group 'Unknown'
            >>> exif.tags = {'Exif': {'Unknown': 0}}
            >>> exif.dump()
            Traceback (most recent call last):
                ...
            ValueError: Invalid tag 'Unknown' in group 'Exif'
        """
        # Copy tags before modifying inplace
        tags = copy.deepcopy(self.tags)
        # Replace key names with codes
        for group in self.tags:
            if group == "thumbnail":
                continue
            if group not in ("0th", "1st", "Exif", "GPS", "Interop"):
                raise ValueError("Invalid group '{0}'".format(group))
            ifd_name = "ImageIFD" if group in ("0th", "1st") else group + "IFD"
            ifd = getattr(piexif, ifd_name)
            for tag in self.tags[group]:
                try:
                    code = getattr(ifd, tag)
                except AttributeError:
                    raise ValueError(
                        "Invalid tag '{0}' in group '{1}'".format(tag, group)
                    )
                tags[group][code] = tags[group].pop(tag)
        # Encode to bytes
        return piexif.dump(tags)

    def insert(self, path: str) -> None:
        """
        Insert :attr:`tags` into an image.

        Arguments:
            path: Path to JPEG or WebP image.

        Example:
            >>> # Copy image file
            >>> import shutil, tempfile
            >>> path = tempfile.NamedTemporaryFile().name
            >>> shutil.copy('tests/AK10b_20141013_020336.JPG', path)
            >>> # Read image exif
            >>> exif = Exif(path)
            >>> exif.iso
            200
            >>> # Edit image exif
            >>> exif.tags['Exif']['ISOSpeedRatings'] = 100
            >>> exif.insert(path)
            >>> Exif(path).iso
            100
        """
        piexif.insert(self.dump(), path)
