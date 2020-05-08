from .imports import np

class uarray(object):

    def __init__(self, mean, sigma=None, **kwargs):
        self.mean = np.asarray(mean, **kwargs)
        if sigma is None:
            sigma = np.zeros(self.mean.shape, dtype=self.mean.dtype)
        elif not np.iterable(sigma):
            sigma = np.full(self.mean.shape, sigma, dtype=self.mean.dtype)
        self.sigma = np.asarray(sigma, **kwargs)
        if self.mean.shape != self.sigma.shape:
            raise ValueError('Means and standard deviations are not the same shape')
        if np.any(self.sigma < 0):
            raise ValueError('Standard deviations cannot be negative')

    @property
    def shape(self):
        return self.mean.shape

    def copy(self):
        return self.__class__(self.mean.copy(), self.sigma.copy())

    def abs(self):
        return self.__abs__()

    def sin(self):
        mean = np.sin(self.mean)
        sigma = np.abs(np.cos(self.mean) * self.sigma)
        return self.__class__(mean, sigma)

    def cos(self):
        mean = np.cos(self.mean)
        sigma = np.abs(np.sin(self.mean) * self.sigma)
        return self.__class__(mean, sigma)

    def tuple(self):
        return self.mean, self.sigma

    def __len__(self):
        return len(self.mean)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.mean == other.mean) & (self.sigma == other.sigma)
        else:
            raise NotImplementedError()

    def __getitem__(self, key):
        return self.__class__(self.mean[key], self.sigma[key])

    def __setitem__(self, key, value):
        if isinstance(value, self.__class__):
            self.mean[key] = value.mean
            self.sigma[key] = value.sigma
        else:
            raise NotImplementedError()

    def __iter__(self):
        return zip(self.mean, self.sigma)

    def __reversed__(self):
        return zip(reversed(self.mean), reversed(self.sigma))

    def __neg__(self):
        return self.__class__(-self.mean, self.sigma.copy())

    def __pos__(self):
        return self.__class__(+self.mean, self.sigma.copy())

    def __abs__(self):
        return self.__class__(np.abs(self.mean), self.sigma.copy())

    def __add__(self, other):
        if isinstance(other, self.__class__):
            mean = self.mean + other.mean
            sigma = np.sqrt(self.sigma**2 + other.sigma**2)
        else:
            mean = self.mean + other
            sigma = self.sigma.copy()
        return self.__class__(mean, sigma)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            mean = self.mean - other.mean
            sigma = np.sqrt(self.sigma**2 + other.sigma**2)
        else:
            mean = self.mean - other
            sigma = self.sigma.copy()
        return self.__class__(mean, sigma)

    def __rsub__(self, other):
        return - self + other

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            mean = self.mean * other.mean
            sigma = np.abs(mean) * np.sqrt(
                (self.sigma / self.mean)**2 + (other.sigma / other.mean)**2)
        else:
            mean = self.mean * other
            sigma = np.abs(other) * self.sigma
        return self.__class__(mean, sigma)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            mean = self.mean / other.mean
            sigma = np.abs(mean) * np.sqrt(
                (self.sigma / self.mean)**2 + (other.sigma / other.mean)**2)
        else:
            return self * (1 / other)
        return self.__class__(mean, sigma)

    def __pow__(self, other):
        if isinstance(other, self.__class__):
            raise NotImplementedError()
        else:
            mean = self.mean**other
            sigma = np.abs((mean * other * self.sigma) / self.mean)
        return self.__class__(mean, sigma)

    def __rtruediv__(self, other):
        return other * self**(-1)
