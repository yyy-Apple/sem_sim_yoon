import pickle

if __name__ == '__main__':
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data)