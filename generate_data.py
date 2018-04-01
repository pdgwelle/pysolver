import pysolver as ps
import pickle

def generate_dataset(n_games=1000, n_moves=15):
	model = ps.Model()
	model.create_training_data(n_games=n_games, n_moves=n_moves)
	with open("models/" + str(n_games) + ".pkl", "wb") as f:
		pickle.dump(model, f)
	return model
