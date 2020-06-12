from .imports import np, scipy, cv2, lmfit, matplotlib, sys, os, warnings, datetime
from . import helpers, config, image

# ---- Controls ----

# Controls (within Cameras) support RANSAC with the following API:
# .size
# .observed(index)
# .predicted(index)


class Points(object):
    """
    `Points` store image-world point correspondences.

    World coordinates (`xyz`) are projected into the camera,
    then compared to the corresponding image coordinates (`uv`).

    Attributes:
        cam (Camera): Camera object
        uv (array): Image coordinates (Nx2)
        xyz (array): World coordinates (Nx3)
        directions (bool): Whether `xyz` are absolute coordinates (False)
            or ray directions (True)
        correction (dict or bool): See `cam.project()`
        size (int): Number of point pairs
        xyz (array): Initial camera position (`cam.xyz`)
        imgsz (array): Initial image size (`cam.imgsz`)
    """

    def __init__(self, cam, uv, xyz, directions=False, correction=False):
        if len(uv) != len(xyz):
            raise ValueError("`uv` and `xyz` have different number of rows")
        self.cam = cam
        self.uv = uv
        self.xyz = xyz
        self.directions = directions
        self.correction = correction
        self.cam_xyz = cam.xyz.copy()
        self.imgsz = cam.imgsz.copy()

    @property
    def size(self):
        return len(self.uv)

    @property
    def cams(self):
        return [self.cam]

    def observed(self, index=None):
        """
        Return observed image coordinates.

        Arguments:
            index (array_like or slice): Indices of points to return, or all if `None`
        """
        if index is None:
            index = slice(None)
        return self.uv[index]

    def predicted(self, index=None):
        """
        Predict image coordinates from world coordinates.

        If the camera position (`cam.xyz`) has changed and `xyz` are ray directions
        (`directions=True`), the point correspondences are invalid and an error is
        raised.

        Arguments:
            index (array_like or slice): Indices of world points to project,
                or all if `None`
        """
        if index is None:
            index = slice(None)
        if self.directions and not self.is_static():
            raise ValueError("Camera has changed position (xyz) and `directions=True`")
        return self.cam.project(
            self.xyz[index], directions=self.directions, correction=self.correction
        )

    def is_static(self):
        """
        Test whether the camera is at its original position.
        """
        return (self.cam.xyz == self.cam_xyz).all()

    def plot(self, index=None, scale=1, width=5, selected="red", unselected=None):
        """
        Plot reprojection errors as quivers.

        Arrows point from observed to predicted coordinates.

        Arguments:
            index (array_like or slice): Indices of points to select, or all if `None`
            scale (float): Scale of quivers
            width (float): Width of quivers
            selected: For selected points, further arguments to
                matplotlib.pyplot.quiver (dict), `None` to hide, or color
            unselected: For unselected points, further arguments to
                matplotlib.pyplot.quiver (dict), `None` to hide, or color
        """
        if index is None:
            index = slice(None)
            other_index = slice(0)
        else:
            other_index = np.delete(np.arange(self.size), index)
        uv = self.observed()
        puv = self.predicted()
        duv = scale * (puv - uv)
        defaults = dict(
            scale=1, scale_units="xy", angles="xy", units="xy", width=width, color="red"
        )
        if unselected is not None:
            if not isinstance(unselected, dict):
                unselected = dict(color=unselected)
            unselected = helpers.merge_dicts(defaults, unselected)
            matplotlib.pyplot.quiver(
                uv[other_index, 0],
                uv[other_index, 1],
                duv[other_index, 0],
                duv[other_index, 1],
                **unselected
            )
        if selected is not None:
            if not isinstance(selected, dict):
                selected = dict(color=selected)
            selected = helpers.merge_dicts(defaults, selected)
            matplotlib.pyplot.quiver(
                uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1], **selected
            )

    def resize(self, size=None, force=False):
        """
        Resize to new image size.

        Resizes both the camera and image coordinates.

        Arguments:
            size: Scale factor relative to the camera's original size (float)
                or target image size (iterable).
                If `None`, image coordinates are resized to fit current
                camera image size.
            force (bool): Whether to use `size` even if it does not preserve
                the original aspect ratio
        """
        if size is not None:
            self.cam.resize(size=size, force=force)
        scale = self.cam.imgsz / self.imgsz
        if any(scale != 1):
            self.uv = self.uv * scale
            self.imgsz = self.cam.imgsz.copy()


class Lines(object):
    """
    `Lines` store image and world lines believed to overlap.

    Image lines (`uvs`) are interpolated to a single array of image points (`uvi`).
    World lines (`xyzs`) are projected into the camera and the nearest point along
    any such lines is matched to each image point.

    Attributes:
        cam (Camera): Camera object
        uvs (iterable): Image line vertices (n, 2)
        uvi (array): Image coordinates interpolated from `uvs` by `step`
        xyzs (iterable): World line vertices (n, 3)
        directions (bool): Whether `xyzs` are absolute coordinates (False)
            or ray directions (True)
        correction (dict or bool): See `cam.project()`
        step (float): Along-line distance between image points
            interpolated from lines `uvs`
        size (int): Number of image points
        xyz (array): Initial camera position (`cam.xyz`)
        imgsz (array): Initial image size (`cam.imgsz`)
    """

    def __init__(self, cam, uvs, xyzs, directions=False, correction=False, step=None):
        self.cam = cam
        # Retain image lines for plotting
        self.uvs = list(uvs)
        self.step = step
        if step:
            self.uvi = np.vstack(
                (helpers.interpolate_line(uv, dx=step) for uv in self.uvs)
            )
        else:
            self.uvi = np.vstack(self.uvs)
        self.xyzs = xyzs
        self.directions = directions
        self.correction = correction
        self.cam_xyz = cam.xyz.copy()
        self.imgsz = cam.imgsz.copy()

    @property
    def size(self):
        return len(self.uvi)

    @property
    def cams(self):
        return [self.cam]

    def observed(self, index=None):
        """
        Return observed image coordinates.

        Arguments:
            index (array_like or slice): Indices of points to return, or all if `None`
        """
        if index is None:
            index = slice(None)
        return self.uvi[index]

    def project(self):
        """
        Project world lines onto the image.

        If the camera position (`cam.xyz`) has changed and `xyz` are ray directions
        (`directions=True`), the point correspondences are invalid and an error is
        raised.

        Returns:
            list: Arrays of image coordinates (Nx2)
        """
        if self.directions and not self.is_static():
            raise ValueError("Camera has changed position (xyz) and `directions=True`")
        xy_step = 1 / self.cam.f.mean()
        uv_edges = self.cam.edges(step=self.cam.imgsz / 2)
        xy_edges = self.cam._image2camera(uv_edges)
        xy_box = np.hstack((np.min(xy_edges, axis=0), np.max(xy_edges, axis=0)))
        puvs = []
        inlines = []
        for xyz in self.xyzs:
            # TODO: Instead, clip lines to 3D polar viewbox before projecting
            # Project world lines to camera
            xy = self.cam._world2camera(
                xyz, directions=self.directions, correction=self.correction
            )
            # Discard nan values (behind camera)
            lines = helpers.boolean_split(xy, np.isnan(xy[:, 0]), include="false")
            for line in lines:
                inlines.append(line)
                # Clip lines in view
                # Resolves coordinate wrap around with large distortion
                for cline in helpers.clip_polyline_box(line, xy_box):
                    # Interpolate clipped lines to ~1 pixel density
                    puvs.append(
                        self.cam._camera2image(
                            helpers.interpolate_line(np.array(cline), dx=xy_step)
                        )
                    )
        if puvs:
            return puvs
        else:
            # If no lines inframe, project line vertices infront
            return [self.cam._camera2image(line) for line in inlines]

    def predicted(self, index=None):
        """
        Return the points on the projected world lines nearest the image coordinates.

        Arguments:
            index (array_like or slice): Indices of image points to include in
                nearest-neighbor search, or all if `None`

        Returns:
            array: Image coordinates (Nx2)
        """
        puv = np.row_stack(self.project())
        distances = helpers.pairwise_distance(self.observed(index=index), puv)
        min_index = np.argmin(distances, axis=1)
        return puv[min_index, :]

    def is_static(self):
        """
        Test whether the camera is at its original position.
        """
        return (self.cam.xyz == self.cam_xyz).all()

    def plot(
        self,
        index=None,
        scale=1,
        width=5,
        selected="red",
        unselected=None,
        observed="green",
        predicted="yellow",
    ):
        """
        Plot the reprojection errors as quivers.

        Arrows point from observed to predicted image coordinates.

        Arguments:
            index (array_like or slice): Indices of points to select, or all if `None`
            scale (float): Scale of quivers
            width (float): Width of quivers
            selected: For selected points, further arguments to
                matplotlib.pyplot.quiver (dict), `None` to hide, or color
            unselected: For unselected points, further arguments to
                matplotlib.pyplot.quiver (dict), `None` to hide, or color
            observed: For image lines, further arguments to
                matplotlib.pyplot.plot (dict), `None` to hide, or color
            predicted: For world lines, further arguments to
                matplotlib.pyplot.plot (dict), `None` to hide, or color
        """
        # Plot image lines
        if observed is not None:
            if not isinstance(observed, dict):
                observed = dict(color=observed)
            observed = helpers.merge_dicts(dict(color="green"), observed)
            for uv in self.uvs:
                matplotlib.pyplot.plot(uv[:, 0], uv[:, 1], **observed)
        # Plot world lines
        if predicted is not None:
            if not isinstance(predicted, dict):
                predicted = dict(color=predicted)
            predicted = helpers.merge_dicts(dict(color="yellow"), predicted)
            puvs = self.project()
            for puv in puvs:
                matplotlib.pyplot.plot(puv[:, 0], puv[:, 1], **predicted)
        # Plot errors
        if selected is not None or unselected is not None:
            if index is None:
                index = slice(None)
            uv = self.observed()
            if not predicted:
                puvs = self.project()
            puv = np.row_stack(puvs)
            distances = helpers.pairwise_distance(uv, puv)
            min_index = np.argmin(distances, axis=1)
            duv = scale * (puv[min_index, :] - uv)
            defaults = dict(
                scale=1,
                scale_units="xy",
                angles="xy",
                units="xy",
                width=width,
                color="red",
            )
            if unselected is not None:
                if not isinstance(unselected, dict):
                    unselected = dict(color=unselected)
                unselected = helpers.merge_dicts(defaults, unselected)
                matplotlib.pyplot.quiver(
                    uv[index, 0],
                    uv[index, 1],
                    duv[index, 0],
                    duv[index, 1],
                    **unselected
                )
            if selected is not None:
                if not isinstance(selected, dict):
                    selected = dict(color=selected)
                selected = helpers.merge_dicts(defaults, selected)
                matplotlib.pyplot.quiver(
                    uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1], **selected
                )

    def resize(self, size=None, force=False):
        """
        Resize to new image size.

        Resizes both the camera and image coordinates.

        Arguments:
            size: Scale factor relative to the camera's original size (float)
                or target image size (iterable).
                If `None`, image coordinates are resized to fit current
                camera image size.
            force (bool): Whether to use `size` even if it does not preserve
                the original aspect ratio
        """
        if size is not None:
            self.cam.resize(size=size, force=force)
        scale = self.cam.imgsz / self.imgsz
        if any(scale != 1):
            for i, uv in enumerate(self.uvs):
                self.uvs[i] = uv * scale
            self.uvi *= scale
            self.imgsz = self.cam.imgsz.copy()


class Matches(object):
    """
    `Matches` store image-image point correspondences.

    The image coordinates (`uvs[i]`) of one camera (`cams[i]`) are projected into the
    other camera (`cams[j]`), then compared to the expected image coordinates for that
    camera (`uvs[j]`).

    Attributes:
        cams (list): Pair of Camera objects
        uvs (list): Pair of image coordinate arrays (n, 2)
        size (int): Number of point pairs
        imgszs (list): Initial image sizes (`cam.imgsz`)
        weights (array): Relative weight of each point pair (n, )
    """

    def __init__(self, cams, uvs, weights=None):
        self.cams = cams
        self.uvs = list(uvs)
        self._test_matches()
        self.imgszs = [cam.imgsz.copy() for cam in cams]
        self.weights = weights

    @property
    def size(self):
        return len(self.uvs[0])

    def _test_matches(self):
        if self.cams[0] is self.cams[1]:
            raise ValueError("Both cameras are the same object")
        uvs = self.uvs
        if uvs is None:
            uvs = getattr(self, "xys", None)
        if len(self.cams) != 2 or len(uvs) != 2:
            raise ValueError("`cams` and coordinate arrays must each have two elements")
        if uvs[0].shape != uvs[1].shape:
            raise ValueError("Coordinate arrays have different shapes")

    def observed(self, index=None, cam=0):
        """
        Return observed image coordinates.

        Arguments:
            index (array_like or slice): Indices of points to return, or all if `None`
            cam (Camera or int): Camera of points to return
        """
        if index is None:
            index = slice(None)
        cam_idx = self.cam_index(cam)
        return self.uvs[cam_idx][index]

    def predicted(self, index=None, cam=0):
        """
        Predict image coordinates for a camera from those of the other camera.

        Arguments:
            index (array_like or slice): Indices of points to project from other camera
            cam (Camera or int): Camera to project points into
        """
        if not self.is_static():
            raise ValueError("Cameras have different positions (xyz)")
        if index is None:
            index = slice(None)
        cam_in = self.cam_index(cam)
        cam_out = 0 if cam_in else 1
        dxyz = self.cams[cam_out].invproject(self.uvs[cam_out][index])
        return self.cams[cam_in].project(dxyz, directions=True)

    def is_static(self):
        """
        Test whether the cameras are at the same position.
        """
        return (self.cams[0].xyz == self.cams[1].xyz).all()

    def cam_index(self, cam):
        """
        Retrieve the index of a camera.

        Arguments:
            cam (Camera): Camera object
        """
        if isinstance(cam, int):
            if cam >= len(self.cams):
                raise IndexError("Camera index out of range")
            return cam
        else:
            return self.cams.index(cam)

    def plot(
        self, index=None, cam=0, scale=1, width=5, selected="red", unselected=None
    ):
        """
        Plot the reprojection errors as quivers.

        Arrows point from the observed to the predicted coordinates.

        Arguments:
            index (array_like or slice): Indices of points to select, or all if `None`
            cam (Camera or int): Camera to plot
            scale (float): Scale of quivers
            width (float): Width of quivers
            selected: For selected points, further arguments to
                matplotlib.pyplot.quiver (dict), `None` to hide, or color
            unselected: For unselected points, further arguments to
                matplotlib.pyplot.quiver (dict), `None` to hide, or color
        """
        if index is None:
            index = slice(None)
            other_index = slice(0)
        else:
            other_index = np.delete(np.arange(self.size), index)
        uv = self.observed(cam=cam)
        puv = self.predicted(cam=cam)
        duv = scale * (puv - uv)
        defaults = dict(
            scale=1, scale_units="xy", angles="xy", units="xy", width=width, color="red"
        )
        if unselected is not None:
            if not isinstance(unselected, dict):
                unselected = dict(color=unselected)
            unselected = helpers.merge_dicts(defaults, unselected)
            matplotlib.pyplot.quiver(
                uv[other_index, 0],
                uv[other_index, 1],
                duv[other_index, 0],
                duv[other_index, 1],
                **unselected
            )
        if selected is not None:
            if not isinstance(selected, dict):
                selected = dict(color=selected)
            selected = helpers.merge_dicts(defaults, selected)
            matplotlib.pyplot.quiver(
                uv[index, 0], uv[index, 1], duv[index, 0], duv[index, 1], **selected
            )

    def as_type(self, mtype):
        """
        Return as a matches object of a different type.
        """
        if mtype is type(self):
            return self
        else:
            return mtype(cams=self.cams, uvs=self.uvs, weights=self.weights)

    def resize(self, size=None, force=False):
        """
        Resize to new image size.

        Resizes both the cameras and their image coordinates.

        Arguments:
            size: Scale factor relative to the cameras' original sizes (float)
                or target image size (iterable).
                If `None`, image coordinates are resized to fit current
                camera image sizes.
            force (bool): Whether to use `size` even if it does not preserve
                the original aspect ratio
        """
        for i, cam in enumerate(self.cams):
            if size is not None:
                cam.resize(size=size, force=force)
            scale = cam.imgsz / self.imgszs[i]
            if any(scale != 1):
                self.uvs[i] = self.uvs[i] * scale
                self.imgszs[i] = cam.imgsz.copy()

    def filter(
        self,
        max_distance=None,
        max_error=None,
        min_weight=None,
        n_best=None,
        scaled=False,
    ):
        selected = np.ones(self.size, dtype=bool)
        if min_weight:
            selected &= self.weights >= min_weight
        if max_distance:
            if scaled:
                max_distance = max_distance * self.cams[0].imgsz.max()
            scale = self.cams[0].imgsz / self.cams[1].imgsz
            distances = np.linalg.norm(
                self.uvs[1][selected] * scale - self.uvs[0][selected], axis=1
            )
            selected[selected] &= distances <= max_distance
        if max_error:
            if scaled:
                max_error = max_error * self.cams[0].imgsz.max()
            errors = np.linalg.norm(
                self.observed(index=selected) - self.predicted(index=selected), axis=1
            )
            selected[selected] &= errors <= max_error
        if n_best:
            weight_order = np.flip(
                np.argsort(self.weights[selected]), axis=0
            )  # descending
            indices = weight_order[: min(n_best, len(weight_order))]
            # NOTE: Switch from boolean to integer indexing
            selected = np.arange(len(selected))[selected][indices]
        if self.uvs is not None:
            self.uvs = [uv[selected] for uv in self.uvs]
        if self.weights is not None:
            self.weights = self.weights[selected]
        # HACK: Support for RotationMatches
        if getattr(self, "xys", None) is not None:
            self.xys = [xy[selected] for xy in self.xys]


class RotationMatches(Matches):
    """
    `RotationMatches` store image-image point correspondences for cameras
    separated by a pure rotation.

    Normalized camera coordinates are pre-computed for speed. Therefore,
    the cameras must always have equal `xyz` (as for `Matches`)
    and no internal parameters can change after initialization.

    Attributes:
        cams (list): Pair of Camera objects
        uvs (list): Pair of image coordinate arrays (Nx2)
        xys (list): Pair of normalized coordinate arrays (Nx2)
        original_internals (list): Original camera internal parameters
            (imgsz, f, c, k, p) of each camera
        size (int): Number of point pairs
    """

    def __init__(self, cams, uvs=None, xys=None, weights=None):
        self.cams = cams
        self.uvs = self._build_uvs(uvs=uvs, xys=xys)
        self.xys = self._build_xys(uvs=uvs, xys=xys)
        self.weights = weights
        self._test_matches()
        # [imgsz, f, c, k, p]
        self.original_internals = [cam.vector.copy()[6:] for cam in self.cams]

    def _build_uvs(self, uvs=None, xys=None):
        if uvs is None and xys is not None:
            return (
                self.cams[0]._camera2image(xys[0]),
                self.cams[1]._camera2image(xys[1]),
            )
        else:
            return uvs

    def _build_xys(self, uvs=None, xys=None):
        if xys is None and uvs is not None:
            return (
                self.cams[0]._image2camera(uvs[0]),
                self.cams[1]._image2camera(uvs[1]),
            )
        else:
            return xys

    def predicted(self, index=None, cam=0):
        """
        Predict image coordinates for a camera from those of the other camera.

        Arguments:
            index (array_like or slice): Indices of points to project from other camera
            cam (Camera or int): Camera to project points into
        """
        if not self.is_static():
            raise ValueError("Cameras have different positions (xyz)")
        if not self.is_original_internals():
            raise ValueError(
                "Camera internal parameters (imgsz, f, c, k, p) have changed"
            )
        if index is None:
            index = slice(None)
        cam_in = self.cam_index(cam)
        cam_out = 0 if cam_in else 1
        dxyz = self.cams[cam_out]._camera2world(self.xys[cam_out][index])
        return self.cams[cam_in].project(dxyz, directions=True)

    def is_original_internals(self):
        """
        Test whether camera internal parameters are unchanged.
        """
        return (
            (self.cams[0].vector[6:] == self.original_internals[0])
            & (self.cams[1].vector[6:] == self.original_internals[1])
        ).all()

    def as_type(self, mtype):
        """
        Return as a matches object of a different type.
        """
        if mtype is type(self):
            return self
        elif mtype is Matches:
            uvs = self._build_uvs(uvs=self.uvs, xys=self.xys)
            return mtype(cams=self.cams, uvs=uvs, weights=self.weights)
        else:
            return mtype(
                cams=self.cams, uvs=self.uvs, xys=self.xys, weights=self.weights
            )


class RotationMatchesXY(RotationMatches):
    """
    `RotationMatchesXY` store image-image point correspondences for cameras
    separated by a pure rotation.

    Normalized camera coordinates are pre-computed for speed,
    and image coordinates may be discarded to save memory (`self.uvs = None`).
    Unlike `RotationMatches`, `self.predicted()` and `self.observed()` return
    normalized camera coordinates.

    Arguments:
        uvs (list): Pair of image coordinate arrays (Nx2)

    Attributes:
        cams (list): Pair of Camera objects
        xys (list): Pair of normalized coordinate arrays (Nx2)
        original_internals (list): Original camera internal parameters
            (imgsz, f, c, k, p) of each camera
        size (int): Number of point pairs
    """

    def __init__(self, cams, uvs=None, xys=None, weights=None):
        self.cams = cams
        self.uvs = uvs
        self.xys = self._build_xys(uvs=uvs, xys=xys)
        self.weights = weights
        self._test_matches()
        # [imgsz, f, c, k, p]
        self.original_internals = [cam.vector.copy()[6:] for cam in self.cams]

    @property
    def size(self):
        return len(self.xys[0])

    def observed(self, index=None, cam=0):
        """
        Return observed camera coordinates.

        Arguments:
            index (array_like or slice): Indices of points to return, or all if `None`
            cam (Camera or int): Camera of points to return
        """
        if index is None:
            index = slice(None)
        cam_idx = self.cam_index(cam)
        return self.xys[cam_idx][index]

    def predicted(self, index=None, cam=0):
        """
        Predict camera coordinates for a camera from those of the other camera.

        Arguments:
            index (array_like or slice): Indices of points to project from other camera
            cam (Camera or int): Camera to project points into
        """
        if not self.is_static():
            raise ValueError("Cameras have different positions (xyz)")
        if not self.is_original_internals():
            raise ValueError(
                "Camera internal parameters (imgsz, f, c, k, p) have changed"
            )
        if index is None:
            index = slice(None)
        cam_in = self.cam_index(cam)
        cam_out = 0 if cam_in else 1
        dxyz = self.cams[cam_out]._camera2world(self.xys[cam_out][index])
        return self.cams[cam_in]._world2camera(dxyz, directions=True)

    def plot(self, *args, **kwargs):
        raise AttributeError("plot() not supported by RotationMatchesXY")


class RotationMatchesXYZ(RotationMatches):
    """
    `RotationMatches3D` store image-image point correspondences for cameras
    separated by a pure rotation.

    Normalized camera coordinates are pre-computed for speed,
    and image coordinates may be discarded to save memory (`self.uvs = None`).
    Unlike `RotationMatches`, `self.predicted()` returns
    world ray directions and `self.observed()` is disabled.

    Arguments:
        uvs (list): Pair of image coordinate arrays (Nx2)

    Attributes:
        cams (list): Pair of Camera objects
        xys (list): Pair of normalized coordinate arrays (Nx2)
        original_internals (list): Original camera internal parameters
            (imgsz, f, c, k, p) of each camera
        size (int): Number of point pairs
    """

    def __init__(self, cams, uvs=None, xys=None, weights=None):
        self.cams = cams
        self.uvs = uvs
        self.xys = self._build_xys(uvs=uvs, xys=xys)
        self.weights = weights
        self._test_matches()
        # [imgsz, f, c, k, p]
        self.original_internals = [cam.vector.copy()[6:] for cam in self.cams]

    @property
    def size(self):
        return len(self.xys[0])

    def observed(self, *args, **kwargs):
        raise AttributeError("observed() not supported by RotationMatchesXYZ")

    def predicted(self, index=None, cam=0):
        """
        Predict world coordinates for a camera.

        Returns world coordinates as ray directions normalized with unit length.

        Arguments:
            index (array_like or slice): Indices of points to project from other camera
            cam (Camera or int): Camera to project points into
        """
        if not self.is_static():
            raise ValueError("Cameras have different positions (xyz)")
        if not self.is_original_internals():
            raise ValueError(
                "Camera internal parameters (imgsz, f, c, k, p) have changed"
            )
        if index is None:
            index = slice(None)
        cam_idx = self.cam_index(cam)
        dxyz = self.cams[cam_idx]._camera2world(self.xys[cam_idx][index])
        # Normalize world coordinates to unit sphere
        dxyz *= 1 / np.linalg.norm(dxyz, ord=2, axis=1).reshape(-1, 1)
        return dxyz

    def plot(self, *args, **kwargs):
        raise AttributeError("plot() not supported by RotationMatchesXY")


# ---- Models ----

# Models support RANSAC with the following API:
# .data_size()
# .fit(index)
# .errors(params, index)


class Polynomial(object):
    """
    Least-squares 1-dimensional polynomial model.

    Fits a polynomial of degree `deg` to 2-dimensional points (rows of `data`) and
    returns the coefficients that minimize the squared error (`params`).
    Can be used with RANSAC algorithm (see optimize.ransac).

    Attributes:
        data (array): Point coordinates (x,y) (Nx2)
        deg (int): Degree of the polynomial
    """

    def __init__(self, data, deg=1):
        self.deg = deg
        self.data = data

    def data_size(self):
        """
        Count the number of points.
        """
        return len(self.data)

    def predict(self, params, index=slice(None)):
        """
        Predict the values of a polynomial.

        Arguments:
            params (array): Values of the polynomial,
                from highest to lowest degree component
            index (array_like or slice): Indices of points for which to predict y from x
        """
        return np.polyval(params, self.data[index, 0])

    def errors(self, params, index=slice(None)):
        """
        Compute the errors of a polynomial prediction.

        Arguments:
            params (array): Values of the polynomial,
                from highest to lowest degree component
            index (array_like or slice): Indices of points for which to predict y from x
        """
        prediction = self.predict(params, index)
        return np.abs(prediction - self.data[index, 1])

    def fit(self, index=slice(None)):
        """
        Fit a polynomial to the points (using numpy.polyfit).

        Arguments:
            index (array_like or slice): Indices of points to use for fitting

        Returns:
            array: Values of the polynomial, from highest to lowest degree component
        """
        return np.polyfit(self.data[index, 0], self.data[index, 1], deg=self.deg)

    def plot(
        self,
        params=None,
        index=slice(None),
        selected="red",
        unselected="grey",
        polynomial="red",
    ):
        """
        Plot the points and the polynomial fit.

        Arguments:
            params (array): Values of the polynomial,
                from highest to lowest degree component, or computed if `None`
            index (array_like or slice): Indices of points to select
            selected (color): Matplotlib color for selected points, or `None` to hide
            unselected (color): Matplotlib color for unselected points,
                or `None` to hide
            polynomial (color): Matplotlib color for polynomial fit, or `None` to hide
        """
        if params is None:
            params = self.fit(index)
        other_index = np.delete(np.arange(self.data_size()), index)
        if selected:
            matplotlib.pyplot.scatter(
                self.data[index, 0], self.data[index, 1], c=selected
            )
        if unselected:
            matplotlib.pyplot.scatter(
                self.data[other_index, 0], self.data[other_index, 1], c=unselected
            )
        if polynomial:
            matplotlib.pyplot.plot(self.data[:, 0], self.predict(params), c=polynomial)


class Cameras(object):
    """
    Multi-camera optimization.

    Finds the camera parameter values that minimize the reprojection errors of camera
    control:

        - image-world point coordinates (Points)
        - image-world line coordinates (Lines)
        - image-image point coordinates (Matches)

    If used with RANSAC (see `optimize.ransac`) with multiple control objects,
    results may be unstable since samples are drawn randomly from all observations,
    and computation will be slow since errors are calculated for all points then subset.

    Arguments:
        scales (bool): Whether to compute and use scale factors for each parameter
        sparsity (bool): Whether compute and use a sparsity structure for the
            estimation of the Jacobian matrix

    Attributes:
        cams (list): Cameras
        controls (list): Camera control (Points, Lines, and Matches objects)
        cam_params (list): Parameters to optimize seperately for each camera
            (see `parse_params()`)
        group_indices (list): Integer index of `cams` belonging to each group
        group_params (list): Parameters to optimize together for all cameras in
            each group (see `parse_params()`)
        weights (array): Weights for each control point
        scales (array): Scale factors for each parameter (see `camera_scales()`)
        sparsity (sparse matrix): Sparsity structure for the estimation of the
            Jacobian matrix
        vectors (list): Original camera vectors
        params (`lmfit.Parameters`): Parameter initial values and bounds
    """

    def __init__(
        self,
        cams,
        controls,
        cam_params=None,
        group_indices=None,
        group_params=None,
        weights=None,
        scales=True,
        sparsity=True,
    ):
        (
            cams,
            controls,
            cam_params,
            group_indices,
            group_params,
        ) = self.__class__._as_lists(
            cams, controls, cam_params, group_indices, group_params
        )
        # Core attributes
        self.cams = cams
        controls = self.__class__.prune_controls(controls, cams=self.cams)
        self.controls = controls
        ncams = len(self.cams)
        if cam_params is None:
            cam_params = [dict()] * ncams
        self.cam_params = cam_params
        if group_indices is None:
            group_indices = [range(ncams)]
        self.group_indices = group_indices
        if group_params is None:
            group_params = [dict()] * len(self.group_indices)
        self.group_params = group_params
        self.weights = weights
        # Build lmfit parameters
        # params, cam_masks, group_masks, cam_breaks, group_breaks for set_cameras()
        self.update_params()
        # Test for errors
        self._test()
        # Save original camera vectors for reset_cameras()
        self.vectors = [cam.vector.copy() for cam in self.cams]
        # Parameter scale factors
        self.scales = None
        if scales:
            self._build_scales()
        # Sparse Jacobian
        self.sparsity = None
        if sparsity:
            self._build_sparsity()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if value is None:
            self._weights = value
        else:
            value = np.atleast_2d(value).reshape(-1, 1)
            self._weights = value * len(value) / sum(value)

    @staticmethod
    def _as_lists(cams, controls, cam_params, group_indices, group_params):
        if isinstance(cams, image.Camera):
            cams = [cams]
        if isinstance(controls, (Points, Lines, Matches)):
            controls = [controls]
        if isinstance(cam_params, dict):
            cam_params = [cam_params]
        if isinstance(group_indices, int):
            group_indices = [group_indices]
        if group_indices is not None and isinstance(group_indices[0], int):
            group_indices = [group_indices]
        if isinstance(group_params, dict):
            group_params = [group_params]
        return cams, controls, cam_params, group_indices, group_params

    @staticmethod
    def _lmfit_labels(mask, cam=None, group=None):
        attributes = ("xyz", "viewdir", "imgsz", "f", "c", "k", "p")
        lengths = [3, 3, 2, 2, 2, 6, 2]
        base_labels = np.array(
            [
                attribute + str(i)
                for attribute, length in zip(attributes, lengths)
                for i in range(length)
            ]
        )
        labels = base_labels[mask]
        if cam is not None:
            labels = ["cam" + str(cam) + "_" + label for label in labels]
        if group is not None:
            labels = ["group" + str(group) + "_" + label for label in labels]
        return labels

    @staticmethod
    def prune_controls(controls, cams):
        """
        Return the controls which reference the specified cameras.

        Arguments:
            controls (list): Camera control (Points, Lines, and Matches)
            cams (list): Camera objects

        Returns:
            list: Control which reference the cameras in `cams`
        """
        return [
            control for control in controls if len(set(cams) & set(control.cams)) > 0
        ]

    @staticmethod
    def camera_scales(cam, controls=None):
        """
        Return camera parameter scale factors.

        These represent the estimated change in each variable in a camera vector needed
        to displace the image coordinates of a feature by one pixel.

        Arguments:
            cam (Camera): Camera object
            controls (list): Camera control (Points, Lines),
                used to estimate impact of camera position (`cam.xyz`).
        """
        # Compute pixels per unit change for each variable
        dpixels = np.ones(20, dtype=float)
        # Compute average distance from image center
        # https://math.stackexchange.com/questions/15580/what-is-average-distance-from-center-of-square-to-some-point
        mean_r_uv = (cam.imgsz.mean() / 6) * (np.sqrt(2) + np.log(1 + np.sqrt(2)))
        mean_r_xy = mean_r_uv / cam.f.mean()
        # xyz (if f is not descaled)
        # Compute mean distance to world features
        if controls:
            means = []
            weights = []
            for control in controls:
                if (
                    isinstance(control, (Points, Lines))
                    and cam is control.cam
                    and not control.directions
                ):
                    weights.append(control.size)
                    if isinstance(control, Points):
                        means.append(np.linalg.norm(control.xyz.mean(axis=0) - cam.xyz))
                    elif isinstance(control, Lines):
                        means.append(
                            np.linalg.norm(
                                np.vstack(control.xyzs).mean(axis=0) - cam.xyz
                            )
                        )
            if means:
                dpixels[0:3] = cam.f.mean() / np.average(means, weights=weights)
        # viewdir[0, 1]
        # First angle rotates camera left-right
        # Second angle rotates camera up-down
        imgsz_degrees = (2 * np.arctan(cam.imgsz / (2 * cam.f))) * (180 / np.pi)
        dpixels[3:5] = cam.imgsz / imgsz_degrees  # pixels per degree
        # viewdir[2]
        # Third angle rotates camera around image center
        theta = np.pi / 180
        dpixels[5] = 2 * mean_r_uv * np.sin(theta / 2)  # pixels per degree
        # imgsz
        dpixels[6:8] = 0.5
        # f (if not descaled)
        dpixels[8:10] = mean_r_xy
        # c
        dpixels[10:12] = 1
        # k (if f is not descaled)
        # Approximate at mean radius
        # NOTE: Not clear why '2**power' terms are needed
        dpixels[12:18] = [
            mean_r_xy ** 3 * cam.f.mean() * 2 ** (1.0 / 2),
            mean_r_xy ** 5 * cam.f.mean() * 2 ** (3.0 / 2),
            mean_r_xy ** 7 * cam.f.mean() * 2 ** (5.0 / 2),
            mean_r_xy ** 3
            / (1 + cam.k[3] * mean_r_xy ** 2)
            * cam.f.mean()
            * 2 ** (1.0 / 2),
            mean_r_xy ** 5
            / (1 + cam.k[4] * mean_r_xy ** 4)
            * cam.f.mean()
            * 2 ** (3.0 / 2),
            mean_r_xy ** 7
            / (1 + cam.k[5] * mean_r_xy ** 6)
            * cam.f.mean()
            * 2 ** (5.0 / 2),
        ]
        # p (if f is not descaled)
        # Approximate at mean radius at 45 degree angle
        dpixels[18:20] = np.sqrt(5) * mean_r_xy ** 2 * cam.f.mean()
        # Convert pixels per change to change per pixel (the inverse)
        return 1 / dpixels

    @staticmethod
    def camera_bounds(cam):
        """
        Return camera parameter bounds.
        """
        # Distortion bounds based on tested limits of Camera.undistort_oulu()
        k = cam.f.mean() / 4000
        p = cam.f.mean() / 40000
        return np.array(
            [
                # xyz
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                # viewdir
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                # imgsz
                [0, np.inf],
                [0, np.inf],
                # f
                [0, np.inf],
                [0, np.inf],
                # c
                [-0.5, 0.5] * cam.imgsz[0:1],
                [-0.5, 0.5] * cam.imgsz[1:2],
                # k
                [-k, k],
                [-k / 2, k / 2],
                [-k / 2, k / 2],
                [-k, k],
                [-k, k],
                [-k, k],
                # p
                [-p, p],
                [-p, p],
            ],
            dtype=float,
        )

    @staticmethod
    def parse_params(params=None, default_bounds=None):
        """
        Return a mask of selected camera parameters and associated bounds.

        Arguments:
            params (dict): Parameters to select by name and indices. For example:

                - {'viewdir': True} : All `viewdir` elements
                - {'viewdir': 0} : First `viewdir` element
                - {'viewdir': [0, 1]} : First and second `viewdir` elements

                Bounds can be specified inside a tuple (indices, min, max).
                Singletons are expanded as needed, and `np.inf` with the
                appropriate sign can be used to indicate no bound:

                - {'viewdir': ([0, 1], -np.inf, 180)}
                - {'viewdir': ([0, 1], -np.inf, [180, 180])}

                `None` or `np.nan` may also be used. These are replaced by the
                values in `default_bounds` (for example, from `camera_bounds()`),
                or (-)`np.inf` if `None`.

        Returns:
            array: Parameter boolean mask (20, )
            array: Parameter min and max bounds (20, 2)
        """
        if params is None:
            params = dict()
        attributes = ("xyz", "viewdir", "imgsz", "f", "c", "k", "p")
        indices = (0, 3, 6, 8, 10, 12, 18, 20)
        mask = np.zeros(20, dtype=bool)
        bounds = np.full((20, 2), np.nan)
        for key, value in params.items():
            if key in attributes:
                if isinstance(value, tuple):
                    selection = value[0]
                else:
                    selection = value
                if selection or selection == 0:
                    i = attributes.index(key)
                    if selection is True:
                        positions = range(indices[i], indices[i + 1])
                    else:
                        positions = indices[i] + np.atleast_1d(selection)
                    mask[positions] = True
                if isinstance(value, tuple):
                    min_bounds = np.atleast_1d(value[1])
                    if len(min_bounds) == 1:
                        min_bounds = np.repeat(min_bounds, len(positions))
                    max_bounds = np.atleast_1d(value[2])
                    if len(max_bounds) == 1:
                        max_bounds = np.repeat(max_bounds, len(positions))
                    bounds[positions] = np.column_stack((min_bounds, max_bounds))
        if default_bounds is not None:
            missing_min = (bounds[:, 0] is None) | (np.isnan(bounds[:, 0]))
            missing_max = (bounds[:, 1] is None) | (np.isnan(bounds[:, 1]))
            bounds[missing_min, 0] = default_bounds[missing_min, 0]
            bounds[missing_max, 1] = default_bounds[missing_max, 1]
        missing_min = (bounds[:, 0] is None) | (np.isnan(bounds[:, 0]))
        missing_max = (bounds[:, 1] is None) | (np.isnan(bounds[:, 1]))
        bounds[missing_min, 0] = -np.inf
        bounds[missing_max, 1] = np.inf
        return mask, bounds

    def _test(self):
        """
        Test for scenarios leading to unexpected results.
        """
        # Error: No controls reference the cameras
        if not len(self.controls):
            raise ValueError("No controls reference the cameras")
        # Error: 'f' or 'c' in `group_params` but image sizes not equal
        for i, idx in enumerate(self.group_indices):
            fc = "f" in self.group_params[i] or "c" in self.group_params[i]
            sizes = np.unique(np.row_stack([self.cams[i].imgsz for i in idx]), axis=0)
            if fc and len(sizes) > 1:
                raise ValueError(
                    "Group "
                    + str(i)
                    + ": 'f' or 'c' in parameters but image sizes not equal"
                )
        # Error: Cameras appear in multiple groups with overlapping masks
        # Test: indices of groups with overlapping masks non-unique
        M = np.row_stack(self.group_masks)
        overlaps = np.nonzero(np.count_nonzero(M, axis=0) > 1)[0]
        for i in overlaps:
            groups = np.nonzero(M[:, i])[0]
            idx = np.concatenate([self.group_indices[group] for group in groups])
            if len(np.unique(idx)) < len(idx):
                raise ValueError(
                    "Some cameras are in multiple groups with overlapping masks"
                )
        # Error: Some cameras with params do not appear in controls
        control_cams = [cam for control in self.controls for cam in control.cams]
        cams_with_params = [
            cam
            for i, cam in enumerate(self.cams)
            if self.cam_params[i]
            or any(
                [
                    self.group_params[j]
                    for j, idx in enumerate(self.group_indices)
                    if i in idx
                ]
            )
        ]
        if set(cams_with_params) - set(control_cams):
            raise ValueError("Not all cameras with params appear in controls")

    def _build_scales(self):
        # TODO: Weigh each camera by number of control points (sum of weights)
        scales = [self.__class__.camera_scales(cam, self.controls) for cam in self.cams]
        cam_scales = [scale[mask] for scale, mask in zip(scales, self.cam_masks)]
        group_scales = [
            np.nanmean(np.row_stack([scales[i][mask] for i in idx]), axis=0)
            for mask, idx in zip(self.group_masks, self.group_indices)
        ]
        self.scales = np.hstack((np.hstack(group_scales), np.hstack(cam_scales)))

    def _build_sparsity(self):
        # Number of observations
        m_control = [2 * control.size for control in self.controls]
        m = sum(m_control)
        # Number of parameters
        n_groups = [np.count_nonzero(mask) for mask in self.group_masks]
        n_cams = [np.count_nonzero(mask) for mask in self.cam_masks]
        n = sum(n_groups) + sum(n_cams)
        # Group lookup
        groups = np.zeros((len(self.cams), len(self.group_indices)), dtype=bool)
        for i, idx in enumerate(self.group_indices):
            groups[idx, i] = True
        # Initialize sparse matrix with zeros
        S = scipy.sparse.lil_matrix((m, n), dtype=int)
        # Build matrix
        control_breaks = np.cumsum([0] + m_control)
        for i, control in enumerate(self.controls):
            ctrl_slice = slice(control_breaks[i], control_breaks[i + 1])
            for cam in control.cams:
                try:
                    j = self.cams.index(cam)
                except ValueError:
                    continue
                cam_slice = slice(self.cam_breaks[j], self.cam_breaks[j + 1])
                # Camera parameters
                S[ctrl_slice, cam_slice] = 1
                # Group parameters
                for group in np.nonzero(groups[j])[0]:
                    group_slice = slice(
                        self.group_breaks[group], self.group_breaks[group + 1]
                    )
                    S[ctrl_slice, group_slice] = 1
        self.sparsity = S

    def update_params(self):
        """
        Update parameter bounds and initial values from current state.
        """
        self.params = lmfit.Parameters()
        # Camera parameters
        cam_bounds = [self.__class__.camera_bounds(cam) for cam in self.cams]
        self.cam_masks, cam_bounds = zip(
            *[
                self.__class__.parse_params(params, default_bounds=bounds)
                for params, bounds in zip(self.cam_params, cam_bounds)
            ]
        )
        cam_labels = [
            self.__class__._lmfit_labels(mask, cam=i, group=None)
            for i, mask in enumerate(self.cam_masks)
        ]
        cam_values = [
            self.cams[i].vector[mask] for i, mask in enumerate(self.cam_masks)
        ]
        # Group parameters
        self.group_masks = []
        for group, idx in enumerate(self.group_indices):
            bounds = np.column_stack(
                (
                    np.column_stack([cam_bounds[i][:, 0] for i in idx]).max(axis=1),
                    np.column_stack([cam_bounds[i][:, 1] for i in idx]).min(axis=1),
                )
            )
            mask, bounds = self.__class__.parse_params(
                self.group_params[group], default_bounds=bounds
            )
            labels = self.__class__._lmfit_labels(mask, cam=None, group=group)
            # NOTE: Initial group values as mean of cameras
            values = np.nanmean(
                np.row_stack([self.cams[i].vector[mask] for i in idx]), axis=0
            )
            for label, value, bound in zip(labels, values, bounds[mask]):
                self.params.add(
                    name=label, value=value, vary=True, min=bound[0], max=bound[1]
                )
            self.group_masks.append(mask)
        # Add camera parameters after group parameters
        for i in range(len(self.cams)):
            for label, value, bound in zip(
                cam_labels[i], cam_values[i], cam_bounds[i][self.cam_masks[i]]
            ):
                self.params.add(
                    name=label, value=value, vary=True, min=bound[0], max=bound[1]
                )
        # Pre-compute index breaks for set_cameras()
        self.group_breaks = np.cumsum(
            [0] + [np.count_nonzero(mask) for mask in self.group_masks]
        )
        self.cam_breaks = np.cumsum(
            [self.group_breaks[-1]]
            + [np.count_nonzero(mask) for mask in self.cam_masks]
        )

    def set_cameras(self, params):
        """
        Set camera parameter values.

        The operation can be reversed with `self.reset_cameras()`.

        Arguments:
            params (iterable or `lmfit.Parameters`): Parameter values ordered first
                by group or camera [group0 | group1 | cam0 | cam1 | ...],
                then ordered by position in `Camera.vector`.
        """
        if isinstance(params, lmfit.parameter.Parameters):
            params = list(params.valuesdict().values())
        for i, idx in enumerate(self.group_indices):
            for j in idx:
                self.cams[j].vector[self.group_masks[i]] = params[
                    self.group_breaks[i] : self.group_breaks[i + 1]
                ]
                self.cams[j].vector[self.cam_masks[j]] = params[
                    self.cam_breaks[j] : self.cam_breaks[j + 1]
                ]

    def reset_cameras(self, vectors=None, save=False):
        """
        Reset camera parameters to their saved values.

        Arguments:
            vectors (iterable): Camera vectors.
                If `None`, the saved vectors are used (`self.vectors`).
            save (bool): Whether to save `vectors` as new defaults.
        """
        if vectors is None:
            vectors = self.vectors
        else:
            if save:
                self.vectors = vectors
        for cam, vector in zip(self.cams, vectors):
            cam.vector = vector.copy()

    def data_size(self):
        """
        Return the total number of data points.
        """
        return np.sum([control.size for control in self.controls])

    def observed(self, index=None):
        """
        Return the observed image coordinates for all camera control.

        See control `observed()` method for more details.

        Arguments:
            index (array or slice): Indices of points to return, or all if `None`
        """
        if len(self.controls) == 1:
            return self.controls[0].observed(index=index)
        else:
            # TODO: Map index to subindices for each control
            if index is None:
                index = slice(None)
            return np.vstack([control.observed() for control in self.controls])[index]

    def predicted(self, params=None, index=None):
        """
        Return the predicted image coordinates for all camera control.

        See control `predicted()` method for more details.

        Arguments:
            params (array or `lmfit.Parameters`): Parameter values
                (see `.set_cameras()`)
            index (array or slice): Indices of points to return,
                or all if `None` (default)
        """
        if params is not None:
            vectors = [cam.vector.copy() for cam in self.cams]
            self.set_cameras(params)
        if len(self.controls) == 1:
            result = self.controls[0].predicted(index=index)
        else:
            # TODO: Map index to subindices for each control
            if index is None:
                index = slice(None)
            result = np.vstack([control.predicted() for control in self.controls])[
                index
            ]
        if params is not None:
            self.reset_cameras(vectors)
        return result

    def residuals(self, params=None, index=None):
        """
        Return the reprojection residuals for all camera control.

        Residuals are the difference between `.predicted()` and `.observed()`.

        Arguments:
            params (array or `lmfit.Parameters`): Parameter values
                (see `.set_cameras()`)
            index (array_like or slice): Indices of points to include, or all if `None`
        """
        d = self.predicted(params=params, index=index) - self.observed(index=index)
        if self.weights is None:
            return d
        else:
            if index is None:
                index = slice(None)
            return d * self.weights[index]

    def errors(self, params=None, index=None):
        """
        Return the reprojection errors for all camera control.

        Errors are the Euclidean distance between `.predicted()` and `.observed()`.

        Arguments:
            params (array or `lmfit.Parameters`): Parameter values
                (see `.set_cameras()`)
            index (array or slice): Indices of points to include, or all if `None`
        """
        return np.linalg.norm(self.residuals(params=params, index=index), axis=1)

    def fit(
        self,
        index=None,
        cam_params=None,
        group_params=None,
        full=False,
        method="least_squares",
        nan_policy="omit",
        reduce_fcn=None,
        **kwargs
    ):
        """
        Return optimal camera parameter values.

        Find the camera parameter values that minimize the reprojection residuals
        or a derivative objective function across all control.
        See `lmfit.minimize()` (https://lmfit.github.io/lmfit-py/fitting.html).

        Arguments:
            index (array or slice): Indices of residuals to include, or all if `None`
            cam_params (list): Sequence of `cam_params` to fit iteratively
                before the final run. Must be `None` or same length as `group_params`.
            group_params (list): Sequence of `group_params` to fit iteratively
                before the final run. Must be `None` or same length as `cam_params`.
            full (bool): Whether to return the full result of `lmfit.Minimize()`
            **kwargs: Additional arguments to `lmfit.minimize()`.
                `self.scales` and `self.jac_sparsity` (if computed) are applied
                to the following arguments if not provided:

                    - `diag=self.scales` for `method='leastsq'`
                    - `x_scale=self.scales` and `jac_sparsity=self.sparsity` for
                    `method='least_squares'`

        Returns:
            array or `lmfit.Parameters` (`full=True`): Parameter values ordered first
                by group or camera (group, cam0, cam1, ...),
                then ordered by position in `Camera.vector`.
        """
        if method == "leastsq":
            if self.scales is not None and not hasattr(kwargs, "diag"):
                kwargs["diag"] = self.scales
        if method == "least_squares":
            if self.scales is not None and not hasattr(kwargs, "x_scale"):
                kwargs["x_scale"] = self.scales
            if self.sparsity is not None and not hasattr(kwargs, "jac_sparsity"):
                if index is None:
                    kwargs["jac_sparsity"] = self.sparsity
                else:
                    if isinstance(index, slice):
                        jac_index = np.arange(self.data_size())[index]
                    else:
                        jac_index = index
                    jac_index = np.dstack((2 * jac_index, 2 * jac_index + 1)).ravel()
                    kwargs["jac_sparsity"] = self.sparsity[jac_index]

        def callback(params, iter, resid, *args, **kwargs):
            err = np.linalg.norm(resid.reshape(-1, 2), ord=2, axis=1).mean()
            sys.stdout.write("\r" + str(err))
            sys.stdout.flush()

        iterations = max(
            len(cam_params) if cam_params else 0,
            len(group_params) if group_params else 0,
        )
        if iterations:
            for n in range(iterations):
                iter_cam_params = cam_params[n] if cam_params else self.cam_params
                iter_group_params = (
                    group_params[n] if group_params else self.group_params
                )
                model = Cameras(
                    cams=self.cams,
                    controls=self.controls,
                    cam_params=iter_cam_params,
                    group_params=iter_group_params,
                )
                values = model.fit(
                    index=index,
                    method=method,
                    nan_policy=nan_policy,
                    reduce_fcn=reduce_fcn,
                    **kwargs
                )
                if values is not None:
                    model.set_cameras(params=values)
            self.update_params()
        result = lmfit.minimize(
            params=self.params,
            fcn=self.residuals,
            kws=dict(index=index),
            iter_cb=callback,
            method=method,
            nan_policy=nan_policy,
            reduce_fcn=reduce_fcn,
            **kwargs
        )
        sys.stdout.write("\n")
        if iterations:
            self.reset_cameras()
            self.update_params()
        if not result.success:
            print(result.message)
        if full:
            return result
        elif result.success:
            return np.array(list(result.params.valuesdict().values()))

    def plot(
        self,
        params=None,
        cam=0,
        index=None,
        scale=1,
        width=5,
        selected="red",
        unselected=None,
        lines_observed="green",
        lines_predicted="yellow",
    ):
        """
        Plot reprojection errors.

        See control object `plot()` methods for details.

        Arguments:
            params (array): Parameter values [group | cam0 | cam1 | ...].
                If `None` (default), cameras are used unchanged.
            cam (Camera or int): Camera to plot in
                (as object or position in `self.cams`)
            index (array or slice): Indices of points to plot.
                If `None` (default), all points are plotted.
                Other values require `self.test_ransac()` to be True.
            scale (float): Scale of quivers
            width (float): Width of quivers
            selected: For selected points,further arguments to
                matplotlib.pyplot.quiver (dict), `None` to hide, or color
            unselected: For unselected points, further arguments to
                matplotlib.pyplot.quiver (dict), `None` to hide, or color
            lines_observed: For image lines, further arguments to
                matplotlib.pyplot.plot (dict), `None` to hide, or color
            lines_predicted: For world lines, further arguments to
                matplotlib.pyplot.plot (dict), `None` to hide, or color
        """
        if index is not None and len(self.controls) > 1:
            # TODO: Map index to subindices for each control
            raise ValueError(
                "Plotting with `index` not yet supported with multiple controls"
            )
        if params is not None:
            vectors = [cam.vector.copy() for cam in self.cams]
            self.set_cameras(params)
        cam = self.cams[cam] if isinstance(cam, int) else cam
        cam_controls = self.__class__.prune_controls(self.controls, cams=[cam])
        for control in cam_controls:
            if isinstance(control, Lines):
                control.plot(
                    index=index,
                    scale=scale,
                    width=width,
                    selected=selected,
                    unselected=unselected,
                    observed=lines_observed,
                    predicted=lines_predicted,
                )
            elif isinstance(control, Points):
                control.plot(
                    index=index,
                    scale=scale,
                    width=width,
                    selected=selected,
                    unselected=unselected,
                )
            elif isinstance(control, Matches):
                control.plot(
                    cam=cam,
                    index=index,
                    scale=scale,
                    width=width,
                    selected=selected,
                    unselected=unselected,
                )
        if params is not None:
            self.reset_cameras(vectors)

    def plot_weights(self, index=None, scale=1, cmap=None):
        if index is None:
            index = slice(None)
        weights = np.ones(self.data_size()) if self.weights is None else self.weights
        uv = self.observed(index=index)
        matplotlib.pyplot.scatter(
            uv[:, 0], uv[:, 1], c=weights[index], s=scale * weights[index], cmap=cmap
        )
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.gca().invert_yaxis()


class ObserverCameras(object):
    """
    `ObserverCameras` finds the optimal view directions of the cameras in an `Observer`.

    Attributes:
        observer (`glimpse.Observer`): Observer with the cameras to orient
        anchors (iterable): Integer indices of `observer.images` to use as anchors.
            If `None`, the first image is used.
        matches (array): Grid of `RotationMatchesXYZ` objects.
        matcher (KeypointMatcher): KeypointMatcher object used by
            `self.build_keypoints()` and `self.build_matches()`
        viewdirs (array): Original camera view directions
    """

    def __init__(self, observer, matches=None, anchors=None):
        self.observer = observer
        if anchors is None:
            is_anchor = [img.anchor for img in self.observer.images]
            anchors = np.where(is_anchor)[0]
            if len(anchors) == 0:
                warnings.warn("No anchor image found, using first image as anchor")
                anchors = (0,)
        self.anchors = anchors
        self.matches = matches
        self.matcher = KeypointMatcher(images=self.observer.images)
        # Placeholders
        self.viewdirs = np.vstack(
            [img.cam.viewdir.copy() for img in self.observer.images]
        )

    def set_cameras(self, viewdirs):
        for i, img in enumerate(self.observer.images):
            img.cam.viewdir = viewdirs[i]

    def reset_cameras(self):
        self.set_cameras(viewdirs=self.viewdirs.copy())

    def build_keypoints(self, *args, **kwargs):
        self.matcher.build_keypoints(*args, **kwargs)

    def build_matches(self, *args, **kwargs):
        self.matcher.build_matches(*args, **kwargs)
        self.matcher.convert_matches(RotationMatchesXYZ)
        self.matches = self.matcher.matches

    def fit(self, anchor_weight=1e6, method="bfgs", **params):
        """
        Return optimal camera view directions.

        The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm is used to find
        the camera view directions that minimize the sum of the absolute differences
        (L1-norm). See `scipy.optimize.minimize(method='bfgs')`.

        Arguments:
            anchor_weight (float): Weight on anchor image view directions being correct
            **params: Additional arguments to `scipy.optimize.minimize()`

        Returns:
            `scipy.optimize.OptimizeResult`: The optimization result.
                Attributes include solution array `x`, boolean `success`, and `message`.
        """
        # Ensure matches are in COO sparse matrix format
        if isinstance(self.matches, scipy.sparse.coo.coo_matrix):
            matches = self.matches
        else:
            matches = scipy.sparse.coo_matrix(matches)

        # Define combined objective, jacobian function
        def fun(viewdirs):
            viewdirs = viewdirs.reshape(-1, 3)
            self.set_cameras(viewdirs=viewdirs)
            objective = 0
            gradients = np.zeros(viewdirs.shape)
            for i in self.anchors:
                objective += (anchor_weight / 2.0) * np.sum(
                    (viewdirs[i] - self.viewdirs[i]) ** 2
                )
                gradients[i] += anchor_weight * (viewdirs[i] - self.viewdirs[i])
            for m, i, j in zip(matches.data, matches.row, matches.col):
                # Project matches
                dxyz = m.predicted(cam=0) - m.predicted(cam=1)
                objective += np.sum(np.abs(dxyz))
                # i -> j
                xy_hat = np.column_stack((m.xys[0], np.ones(m.size)))
                dD_dw = np.matmul(m.cams[0].Rprime, xy_hat.T)
                delta = np.sign(dxyz).reshape(-1, 3, 1)
                gradient = np.sum(np.matmul(dD_dw.T, delta).T, axis=2).squeeze()
                gradients[i] += gradient
                # j -> i
                gradients[j] -= gradient
            # Update console output
            sys.stdout.write("\r" + str(objective))
            sys.stdout.flush()
            return objective, gradients.ravel()

        # Optimize camera view directions
        viewdirs_0 = [img.cam.viewdir for img in self.observer.images]
        result = scipy.optimize.minimize(
            fun=fun, x0=viewdirs_0, jac=True, method=method, **params
        )
        self.reset_cameras()
        if not result.success:
            sys.stdout.write("\n")  # new line
            print(result.message)
        return result


# ---- RANSAC ----


def ransac(model, sample_size, max_error, min_inliers, iterations=100, **fit_kws):
    """
    Fit model parameters to data using the Random Sample Consensus (RANSAC) algorithm.

    Inspired by the pseudocode at https://en.wikipedia.org/wiki/Random_sample_consensus

    Arguments:
        model (object): Model and data object with the following methods:

            - `data_size()`: Returns maximum sample size
            - `fit(index)`: Accepts sample indices and returns model parameters
            - `errors(params, index)`: Accepts sample indices and model parameters and
                returns an error for each sample

        sample_size (int): Size of sample used to fit the model in each iteration
        max_error (float): Error below which a sample element is considered
            a model inlier
        min_inliers (int): Number of inliers (in addition to `sample_size`) for a model
            to be considered valid
        iterations (int): Number of iterations
        **fit_kws: Additional arguments to `model.fit()`

    Returns:
        array (int): Values of model parameters
        array (int): Indices of model inliers
    """
    i = 0
    params = None
    err = np.inf
    inlier_idx = None
    while i < iterations:
        maybe_idx, test_idx = ransac_sample(sample_size, model.data_size())
        # maybe_inliers = data[maybe_idx]
        maybe_params = model.fit(maybe_idx, **fit_kws)
        if maybe_params is None:
            continue
        # test_data = data[test_idx]
        test_errs = model.errors(maybe_params, test_idx)
        also_idx = test_idx[test_errs < max_error]
        if len(also_idx) > min_inliers:
            # also_inliers = data[also_idx]
            better_idx = np.concatenate((maybe_idx, also_idx))
            better_params = model.fit(better_idx, **fit_kws)
            if better_params is None:
                continue
            better_errs = model.errors(better_params, better_idx)
            this_err = np.mean(better_errs)
            if this_err < err:
                params = better_params
                err = this_err
                inlier_idx = better_idx
        i += 1
    if params is None:
        raise ValueError("Best fit does not meet acceptance criteria")
    # HACK: Recompute inlier index on best params
    inlier_idx = np.where(model.errors(params) <= max_error)[0]
    return params, inlier_idx


def ransac_sample(sample_size, data_size):
    """
    Generate index arrays for a random sample and its outliers.

    Arguments:
        sample_size (int): Size of sample
        data_size (int): Size of data

    Returns:
        array (int): Sample indices
        array (int): Outlier indices
    """
    if sample_size >= data_size:
        raise ValueError("`sample_size` is larger or equal to `data_size`")
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    return indices[:sample_size], indices[sample_size:]


# ---- Keypoints ----


def detect_keypoints(array, mask=None, method="sift", root=True, **params):
    """
    Return keypoints and descriptors for an image.

    Arguments:
        array (array): 2 or 3-dimensional image array (uint8)
        mask (array): Regions in which to detect keypoints (uint8)
        root (bool): Whether to return square root L1-normalized descriptors.
            See https://doi.org/10.1109/CVPR.2012.6248018.
        **params: Additional arguments passed to `cv2.xfeatures2d.SIFT()` or
            `cv2.xfeatures2d.SURF()`.
            See https://docs.opencv.org/master/d2/dca/group__xfeatures2d__nonfree.html.

    Returns:
        list: Keypoints as cv2.KeyPoint objects
        array: Descriptors as array rows
    """
    if method == "sift":
        try:
            detector = cv2.xfeatures2d.SIFT_create(**params)
        except AttributeError:
            # OpenCV 2
            detector = cv2.SIFT(**params)
    elif method == "surf":
        try:
            detector = cv2.xfeatures2d.SURF_create(**params)
        except AttributeError:
            # OpenCV 2
            detector = cv2.SURF(**params)
    keypoints, descriptors = detector.detectAndCompute(array, mask=mask)
    # Empty result: ([], None)
    if root and descriptors is not None:
        descriptors *= 1 / (descriptors.sum(axis=1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors)
    return keypoints, descriptors


def match_keypoints(
    ka,
    kb,
    mask=None,
    max_ratio=None,
    max_distance=None,
    indexParams=dict(algorithm=1, trees=5),
    searchParams=dict(checks=50),
    return_ratios=False,
):
    """
    Return the coordinates of matched keypoint pairs.

    Arguments:
        ka (tuple): Keypoints of image A (keypoints, descriptors)
        kb (tuple): Keypoints of image B (keypoints, descriptors)
        mask (array): Region in which to retain keypoints (uint8)
        max_ratio (float): Maximum descriptor-distance ratio between the best and
            second best match.
            See http://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf#page=20.
        max_distance (float): Maximum coordinate-distance of matched keypoints
        indexParams (dict): Undocumented argument passed to `cv2.FlannBasedMatcher()`
        searchParams (dict): Undocumented argument passed to `cv2.FlannBasedMatcher()`
        return_ratios (bool): Whether to return the ratio of each (filtered) match

    Returns:
        array: Coordinates of matches in image A (n, 2)
        array: Coordinates of matches in image B (n, 2)
        array (optional): Ratio of each match (n, )
    """
    compute_ratios = max_ratio or return_ratios
    n_nearest = 2 if compute_ratios else 1
    if len(ka[0]) >= n_nearest and len(kb[0]) >= n_nearest:
        flann = cv2.FlannBasedMatcher(
            indexParams=indexParams, searchParams=searchParams
        )
        matches = flann.knnMatch(ka[1], kb[1], k=n_nearest, mask=mask)
        uvA = np.array([ka[0][m[0].queryIdx].pt for m in matches]).reshape(-1, 2)
        uvB = np.array([kb[0][m[0].trainIdx].pt for m in matches]).reshape(-1, 2)
        if compute_ratios:
            ratios = np.array([m.distance / n.distance for m, n in matches])
        if max_ratio:
            is_valid = (
                np.array([m.distance / n.distance for m, n in matches]) < max_ratio
            )
            uvA = uvA[is_valid]
            uvB = uvB[is_valid]
            if return_ratios:
                ratios = ratios[is_valid]
        if max_distance:
            is_valid = np.linalg.norm(uvA - uvB, axis=1) < max_distance
            uvA = uvA[is_valid]
            uvB = uvB[is_valid]
            if return_ratios:
                ratios = ratios[is_valid]
    else:
        # Not enough keypoints to match
        empty = np.array([], dtype=float).reshape(-1, 2)
        uvA, uvB = empty, empty.copy()
        ratios = np.array([], dtype=float)
    if return_ratios:
        return uvA, uvB, ratios
    else:
        return uvA, uvB


class KeypointMatcher(object):
    """
    `KeypointMatcher` detects and matches image keypoints.

        - Build (and save to file) keypoint descriptors for each image with
            `self.build_keypoints()`.
        - Build (and save to file) keypoints matches between image pairs with
            `self.build_matches()`.

    Arguments:
        clahe: Arguments to `cv2.createCLAHE()` (dict: clipLimit, tileGridSize)
            or whether to use CLAHE (bool).

    Attributes:
        images (array): Image objects in ascending temporal order
        clahe (cv2.CLAHE): CLAHE object
        matches (scipy.sparse.coo.coo_matrix): Sparse matrix of image-image `Matches`
        keypoints (list): List of image keypoints
            (see `optimize.detect_keypoints()`).
    """

    def __init__(self, images, clahe=False):
        dts = np.diff([img.datetime for img in images])
        if np.any(dts < datetime.timedelta(0)):
            raise ValueError("Images are not in ascending temporal order")
        self.images = np.asarray(images)
        if clahe is False:
            self.clahe = None
        else:
            if clahe is True:
                clahe = dict()
            self.clahe = cv2.createCLAHE(**clahe)
        # Placeholders
        self.keypoints = None
        self.matches = None

    def _prepare_image_basenames(self):
        basenames = [helpers.strip_path(img.path) for img in self.images]
        if len(basenames) != len(set(basenames)):
            raise ValueError("Image basenames are not unique")
        return basenames

    def _prepare_image(self, I):
        """
        Prepare image data for keypoint detection.
        """
        if I.ndim > 2:
            I = helpers.rgb_to_gray(I, method="average", weights=None)
        if self.clahe is not None:
            I = self.clahe.apply(I.astype(np.uint8))
        return I.astype(np.uint8)

    def build_keypoints(
        self,
        masks=None,
        path=None,
        overwrite=False,
        clear_images=True,
        clear_keypoints=False,
        parallel=False,
        **params
    ):
        """
        Build image keypoints and their descriptors.

        Results are stored in `self.keypoints` and/or written to a binary
        `pickle` file with name `basename.pkl`.

        Arguments:
            masks (iterable): Boolean array(s) (uint8) indicating regions in which to
                detect keypoints
            path (str): Directory path for keypoint files.
                If `None`, no files are written.
            overwrite (bool): Whether to recompute and overwrite existing file or
                in-memory keypoints.
            clear_images (bool): Whether to clear cached image data
                (`self.images[i].I`).
            clear_keypoints (bool): Whether to clear cached keypoints
                (`self.keypoints[i]`).
            parallel: Number of image keypoints to detect in parallel (int),
                or whether to detect in parallel (bool). If `True`,
                defaults to `os.cpu_count()`.
            **params: Additional arguments to `optimize.detect_keypoints()`
        """
        if clear_keypoints and not path:
            raise ValueError("path is required when clear_keypoints is True")
        if path and os.path.isfile(path):
            raise ValueError("path must be a directory")
        basenames = self._prepare_image_basenames()
        # Enforce defaults
        if masks is None or isinstance(masks, np.ndarray):
            masks = (masks,) * len(self.images)
        parallel = helpers._parse_parallel(parallel)
        if not self.keypoints:
            self.keypoints = [None] * len(self.images)

        # Define parallel process
        def process(i, img):
            print(img.path)
            if path:
                outpath = os.path.join(path, basenames[i] + ".pkl")
                written = os.path.isfile(outpath)
            else:
                written = False
            keypoints = self.keypoints[i]
            read = keypoints is not None
            if not read and written and not clear_keypoints:
                # Read keypoints from file
                keypoints = helpers.read_pickle(outpath)
            elif read and not written and path:
                # Write keypoints to file
                helpers.write_pickle(keypoints, path=outpath)
            elif (not read and not written) or overwrite:
                # Detect keypoints
                I = self._prepare_image(img.read())
                keypoints = detect_keypoints(I, mask=masks[i], **params)
                if path:
                    # Write keypoints to file
                    helpers.write_pickle(keypoints, path=outpath)
                if clear_images:
                    img.I = None
            if clear_keypoints:
                keypoints = None
            return keypoints

        # Run process in parallel
        with config._MapReduce(np=parallel) as pool:
            self.keypoints = pool.map(
                func=process, sequence=tuple(enumerate(self.images)), star=True
            )

    def build_matches(
        self,
        maxdt=None,
        min_nearest=0,
        seq=None,
        imgs=None,
        keypoints_path=None,
        path=None,
        overwrite=False,
        clear_keypoints=True,
        clear_matches=False,
        parallel=False,
        weights=False,
        as_type=None,
        filter=None,
        **params
    ):
        """
        Build matches between each image and its nearest neighbors.

        Results are stored in `self.matches` as an (n, n) upper-triangular sparse matrix
        of `Matches`, and the result for each `Image` pair (i, j) optionally written to
        a binary `pickle` file with name `basenames[i]-basenames[j].pkl`. If
        `clear_matches` is `True`, missing files are written but results are not stored
        in memory.

        Arguments:
            maxdt (`datetime.timedelta`): Maximum time separation between
                pairs of images to match. If `None`, all pairs are matched.
            min_nearest (int): Minimum nearest neighbors to match on either side
                of image (overrides `maxdt`)
            seq (iterable): Positive index of neighbors to match to each image
                (relative to 0). Is in addition to `maxdt` and `min_nearest`.
            imgs (iterable): Index of images to require at least one of
                in each matched image pair. If `None`, all image pairs meeting
                the criteria are matched.
            keypoints_path (str): Directory with keypoint files.
            path (str): Directory for match files. If `None`, no files are written.
            overwrite (bool): Whether to recompute and overwrite existing match files
            clear_keypoints (bool): Whether to clear cached keypoints
                (`self.keypoints`)
            clear_matches (bool): Whether to clear matches rather than return them
                (requires `path`). Useful for avoiding memory overruns when
                processing very large image sets.
            parallel: Number of image keypoints to detect in parallel (int),
                or whether to detect in parallel (bool). If `True`,
                defaults to `os.cpu_count()`.
            weights (bool): Whether to include weights in `Matches` objects,
                computed as the inverse of the maximum descriptor-distance ratio
            filter (dict): Arguments to `optimize.Matches.filter()`.
                If truthy, `Matches` are filtered before being saved to `self.matches`.
                Ignored if `clear_matches=True`.
            **params: Additional arguments to `optimize.match_keypoints()`
        """
        if clear_matches and not path:
            raise ValueError("path is required when clear_keypoints is True")
        if path and os.path.isfile(path):
            raise ValueError("path must be a directory")
        parallel = helpers._parse_parallel(parallel)
        params = helpers.merge_dicts(params, dict(return_ratios=weights))
        basenames = self._prepare_image_basenames()
        if self.keypoints is None:
            self.keypoints = [None] * len(self.images)
        # Match images
        n = len(self.images)
        if maxdt is None:
            matching_images = [np.arange(i + 1, n) for i in range(n)]
        else:
            datetimes = np.array([img.datetime for img in self.images])
            ends = np.searchsorted(datetimes, datetimes + maxdt, side="right")
            if min_nearest:
                shift = min(min_nearest, n) + 1
                min_ends = [min(i + shift, n) for i in range(n)]
                ends = np.maximum(ends, min_ends)
            matching_images = [np.arange(i + 1, end) for i, end in enumerate(ends)]
        # Add match sequence
        if seq is not None:
            seq = np.asarray(seq)
            seq = np.unique(seq[seq > 0])
            for i, m in enumerate(matching_images):
                iseq = seq + i
                iseq = iseq[: np.searchsorted(iseq, n)]
                matching_images[i] = np.unique(np.concatenate((m, iseq)))
        # Filter matched image pairs
        if imgs is not None:
            for i, m in enumerate(matching_images):
                matching_images[i] = m[np.isin(m, imgs)]

        # Define parallel process
        def process(i, js):
            if len(js) > 0:
                print("Matching", i, "->", ", ".join(js.astype(str)))
            matches = []
            imgA = self.images[i]
            if self.keypoints[i] is None and keypoints_path:
                self.keypoints[i] = helpers.read_pickle(
                    os.path.join(keypoints_path, basenames[i] + ".pkl")
                )
            for j in js:
                imgB = self.images[j]
                if self.keypoints[j] is None and keypoints_path:
                    self.keypoints[j] = helpers.read_pickle(
                        os.path.join(keypoints_path, basenames[j] + ".pkl")
                    )
                if path:
                    outfile = os.path.join(
                        path, basenames[i] + "-" + basenames[j] + ".pkl"
                    )
                if path and not overwrite and os.path.exists(outfile):
                    if not clear_matches:
                        match = helpers.read_pickle(outfile)
                        # Point matches to existing Camera objects
                        match.cams = (imgA.cam, imgB.cam)
                        if as_type:
                            match = match.as_type(as_type)
                        matches.append(match)
                else:
                    result = match_keypoints(
                        self.keypoints[i], self.keypoints[j], **params
                    )
                    match = Matches(
                        cams=(imgA.cam, imgB.cam),
                        uvs=result[0:2],
                        weights=(1 / result[2]) if weights else None,
                    )
                    if path is not None:
                        helpers.write_pickle(match, outfile)
                    if not clear_matches:
                        if as_type:
                            match = match.as_type(as_type)
                        matches.append(match)
            if clear_keypoints:
                self.keypoints[i] = None
            return None if clear_matches else matches

        def reduce(matches):
            if filter:
                for match in matches:
                    if match:
                        match.filter(**filter)
            return matches

        # Run process in parallel
        with config._MapReduce(np=parallel) as pool:
            matches = pool.map(
                func=process,
                reduce=reduce,
                star=True,
                sequence=tuple(enumerate(matching_images)),
            )
            if not clear_matches:
                # Build Compressed Sparse Row (CSR) matrix
                matches = scipy.sparse.csr_matrix(
                    (
                        np.concatenate(matches),  # data
                        np.concatenate(matching_images),  # column indices
                        np.cumsum([0] + [len(row) for row in matching_images]),
                    )
                )  # row ranges
                # Convert to Coordinate Format (COO) matrix
                matches = matches.tocoo()
        if clear_matches:
            self.matches = None
        else:
            self.matches = matches
            if parallel:
                self._assign_cameras()

    def _test_matches(self):
        if self.matches is None:
            raise ValueError("Matches have not been initialized. Run build_matches()")

    def _assign_cameras(self):
        for m, i, j in zip(self.matches.data, self.matches.row, self.matches.col):
            m.cams = self.images[i].cam, self.images[j].cam

    def convert_matches(self, mtype, clear_uvs=False, parallel=False):
        self._test_matches()
        parallel = helpers._parse_parallel(parallel)

        def process(i, m):
            m = m.as_type(mtype)
            if clear_uvs and mtype in (RotationMatchesXY, RotationMatchesXYZ):
                m.uvs = None
            return i, m

        def reduce(i, m):
            self.matches.data[i] = m

        with config._MapReduce(np=parallel) as pool:
            _ = pool.map(
                func=process,
                reduce=reduce,
                star=True,
                sequence=tuple(zip(range(self.matches.data.size), self.matches.data)),
            )
        if parallel:
            self._assign_cameras()

    def filter_matches(self, clear_weights=False, parallel=False, **params):
        self._test_matches()
        parallel = helpers._parse_parallel(parallel)

        def process(i, m):
            if params:
                m.filter(**params)
            if clear_weights:
                m.weights = None
            return i, m

        def reduce(i, m):
            self.matches.data[i] = m

        with config._MapReduce(np=parallel) as pool:
            _ = pool.map(
                func=process,
                reduce=reduce,
                star=True,
                sequence=tuple(zip(range(self.matches.data.size), self.matches.data)),
            )
        if parallel:
            self._assign_cameras()

    def _images_mask(self, imgs):
        if np.iterable(imgs):
            return np.isin(self.matches.row, imgs) | np.isin(self.matches.col, imgs)
        else:
            return (self.matches.row == imgs) | (self.matches.col == imgs)

    def _images_matches(self, imgs):
        mask = self._images_mask(imgs)
        return self.matches.data[mask]

    def matches_per_image(self):
        self._test_matches()
        image_matches = [self._images_matches(i) for i in range(len(self.images))]
        # n_images = np.array([
        #   np.sum([mi.size > 0 for mi in m]) for m in image_matches
        # ])
        return np.array([np.sum([mi.size for mi in m]) for m in image_matches])

    def images_per_image(self):
        self._test_matches()
        image_matches = [self._images_matches(i) for i in range(len(self.images))]
        return np.array([np.sum([mi.size > 0 for mi in m]) for m in image_matches])

    def drop_images(self, imgs):
        self._test_matches()
        mask = self._images_mask(imgs)
        self.matches.data[mask] = False
        self.matches.eliminate_zeros()
        # Find all images with no matches
        all = np.arange(len(self.images))
        keep = np.union1d(self.matches.row, self.matches.col)
        drop = np.setdiff1d(all, keep)
        # Remove row and column of each dropped image
        _, new_row = np.unique(
            np.concatenate((self.matches.row, keep)), return_inverse=True
        )
        self.matches.row = new_row[: -len(keep)]
        _, new_col = np.unique(
            np.concatenate((self.matches.col, keep)), return_inverse=True
        )
        self.matches.col = new_col[: -len(keep)]
        # Resize matches matrix
        n = len(self.images) - len(drop)
        self.matches._shape = (n, n)
        # Remove dropped images
        self.images = np.delete(self.images, drop)

    def match_breaks(self, min_matches=0):
        self._test_matches()
        all_starts = np.arange(len(self.images) - 1)
        starts, counts = np.unique(self.matches.row, return_counts=True)
        breaks = np.setdiff1d(all_starts, starts)
        if min_matches:
            min_matches = np.minimum(
                min_matches, len(self.images) - np.arange(len(self.images))
            )
            breaks = np.sort(
                np.concatenate((breaks, np.where(counts < min_matches)[0]))
            )
        return breaks
