class FileAddedObservable:
    def __init__(self):
        self._observers = []

    def register_observer(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def notify_file_added(self):
        for observer in self._observers:
            observer.on_file_added()
