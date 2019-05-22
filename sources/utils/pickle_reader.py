import pickle

pickle_in = open("models/classification_third.pickle","rb")
example_dict = pickle.load(pickle_in)
print(example_dict)
