def main(argv=None):
    from bmpt.cli.train import main as _main

    return _main(argv)


__all__ = ["main"]
