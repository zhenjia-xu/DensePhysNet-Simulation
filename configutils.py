import argparse

class Config(object):
    def __init__(self, name: str):
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, '_config', dict())

    def __setattr__(self, key, value):
        self._config.update({key: value})

    def __getattr__(self, item):
        if item in self._config:
            return self._config[item]
        raise AttributeError('No attribute {}'.format(item))

    def set_parser(self, parser: argparse.ArgumentParser, prefix=''):
        for key, value in self._config.items():
            name = prefix + '_' + key if prefix else key
            if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                print('before' + parser)
                parser.add_argument('--' + name.replace('_', '-'), default=value, type=type(value))
                print(parser)
            elif isinstance(value, Config):
                value.set_parser(parser, key)

    def update(self, config: argparse.Namespace, prefix=''):
        config_dict = dict()
        for k, v in self._config.items():
            name = prefix + '_' + k if prefix else k
            if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                config_dict.setdefault(k, getattr(config, name))
            elif isinstance(v, Config):
                v.update(config, k)
        self._config.update(config_dict)

    def __str__(self):
        contents = []
        for k, v in self._config.items():
            contents.append('{}='.format(k) + ('"{}"'.format(v) if isinstance(v, str) else str(v)))
        return self.name.capitalize() + 'Config({})'.format(",\n".join(contents))

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        result = {}
        for k, v in self._config.items():
            if isinstance(v, Config):
                result.setdefault(k, v.to_dict())
            else:
                result.setdefault(k, v)
        return result