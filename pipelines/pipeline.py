class Pipeline(object):
    def __init__(self, handlers=None):
        self.handlers = [] if handlers is None else handlers
        prev = None
        for handler in self.handlers:
            handler._set_prev(prev)
            handler.set_output_spec()
            prev = handler

    def add_handler(self, handler):
        self.handlers.append(handler)

    def del_handler(self, idx):
        del self.handlers[idx]

    def process(self, dataset=None):
        prev = None
        for handler in self.handlers:
            handler._set_prev(prev)
            dataset = handler.handle(dataset)
            prev = handler
        return dataset
