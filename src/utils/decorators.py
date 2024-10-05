def augment_func(name):
    def decorator(func):
        func.is_augment_func = True
        func.call_name = name
        return func
    return decorator