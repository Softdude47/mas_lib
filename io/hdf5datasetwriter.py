import h5py

class HDF5DatasetWriter:
    def __init__(self, file_name: str, feature_ref_name: str, buffer_size: int, shape: list or tuple):
        
        # hdf file/database with sub directories of labels and data
        self.db = h5py.File(file_name, mode="w")
        self.features = self.db.create_dataset(name=feature_ref_name, shape=shape, dtype=float)
        self.labels = self.db.create_dataset(name="labels", shape=(shape[0], ), dtype=int)
        
        # buffer
        self.buffer = {"features": [], "labels": []}
        self.buffer_size = buffer_size
        self.index = 0
    
            
    def __flush(self,):
        """ moves data in buffer to storage """
        # get current index
        current_index = self.index + len(self.buffer["features"])
        
        # store dataset's features and label using index
        self.features[self.index : current_index] = self.buffer["features"]
        self.labels[self.index : current_index] = self.buffer["labels"]
        
        # update index and clear buffer
        self.index = current_index
        self.buffer = {"features": [], "labels": []}
    
    def add(self, features: list, labels: list):
        """ registers data to buffer """
        self.buffer["features"].extend(features)
        self.buffer["labels"].extend(labels)
        
        if len(self.buffer["features"]) > self.buffer_size:
            self.__flush()
    def close(self,):
        """ closes database/file """
        if len(self.buffer["features"]) > 0:
            self.__flush()
            
        self.db.close()
    
    def store_class_label(self, class_labels):
        dtype = h5py.special_dtype(vlen=str)
        labels_dataset = self.db.create_dataset(name="class_labels", shape=(len(class_labels),), dtype=dtype)
        labels_dataset[:] = class_labels