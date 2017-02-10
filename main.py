import pysolver as ps

model = ps.Model()
model.create_training_data(n_games=1000, n_moves=20)
model.train_model()

c = ps.Cube()
print(c.scramble_cube(3))
original_cube, finished_cube, iterations, prediction_set = c.solve(model)