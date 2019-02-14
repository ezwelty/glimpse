if __name__ == "__main__":
    import doctest
    import glimpse
    modules = [attr for attr in dir(glimpse) if attr[0].islower()]
    for module in modules:
        doctest.testmod(getattr(glimpse, module), verbose=True,
            optionflags=doctest.ELLIPSIS)
