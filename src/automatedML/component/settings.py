

class BaseSettings():

    def __init__(self,**kargs) -> None:
        self.load(**kargs)

    def to_dict(self):
        o = {}
        for name in self.get_fields():
            value = getattr(self, name)
            if value is None:
                o[name] = None
            elif isinstance(value, BaseSettings):
                o[name] = value.to_dict()
            elif isinstance(value, type):
                o[name] = value.__dict__
            else:
                o[name] = value
        return o

    def get_fields(self):
        return [x for x in dir(self.__class__) if not x.startswith('_') and not x in ["get_fields", "clone", "load","to_dict"]]

    def clone(self):
        opt = self.__class__()
        for key in self.get_fields():
            value = getattr(self, key)
            if isinstance(value, BaseSettings):
                setattr(opt, key, value.clone())
            else:
                setattr(opt, key)
        return opt

    def load(self, **kargs) -> None:
        if kargs is not None:
            allowed_keys = self.get_fields()
            for key, val in kargs.items():
                if key in allowed_keys:
                    current_val = getattr(self, key)
                    if current_val is not None and isinstance(current_val, BaseSettings) and isinstance(val, dict):
                        current_val.load(**val)
                    else:
                        setattr(self, key, val)
