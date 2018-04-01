import pycuber as pc

import numpy as np
import pandas as pd

import copy
import time

from sklearn.neural_network import MLPRegressor

class Cube(pc.Cube):

    def is_completed(self):
        sides = ['F', 'U', 'R', 'D', 'L', 'B']
        completed_boolean_array = []
        for side in sides:
            face = Face(self, side)
            completed_boolean_array.append(face.is_completed())

        return all(completed_boolean_array)

    def random_move(self, moves = ['F', 'U', 'R', 'D', 'L', 'B', 'F\'', 'U\'', 'R\'', 'D\'', 'L\'', 'B\'']):
        move = np.random.choice(moves)
        self.perform_step(move)
        return move

    def scramble_cube(self, n_moves=1000):
        move_list = []
        move_string = ''
        move_formula = pc.Formula(move_string)
        while(len(move_list) < n_moves):
            move = self.random_move()
            move_string = move_string + ' ' + move
            move_formula = pc.Formula(move_string).optimise()
            move_list = move_formula.__repr__().split(" ")
        output_moves = copy.deepcopy(move_formula)
        solution = move_formula.reverse()
        return output_moves, solution

    def get_percent_completed(self):
        percent_completed = []
        sides = ['L', 'U', 'F', 'D', 'R', 'B']
        face_list = [Face(self, side) for side in sides]
        for face in face_list:
            percent_completed.append(face.get_percent_completed())
        return np.average(percent_completed)

    def get_completed_sides(self):
        completed_sides = []
        faces = ['L', 'U', 'F', 'D', 'R', 'B']
        face_list = [Face(self, side) for side in sides]
        for face in face_list:
            completed_sides.append(face.is_completed())
        return np.sum(completed_sides)

    def get_completed_pieces(self):
        out_list = []
        out_list.extend(Face(self, 'F').get_dummies(check_val = 'corners'))
        out_list.extend(Face(self, 'B').get_dummies(check_val = 'corners'))
        out_list.extend(Face(self, 'F').get_dummies(check_val = 'centers'))
        out_list.extend(Face(self, 'B').get_dummies(check_val = 'centers'))
        L_centers = Face(self, 'L').get_dummies(check_val = 'centers')
        out_list.extend([L_centers[0], L_centers[2]])
        R_centers = Face(self, 'R').get_dummies(check_val = 'centers')
        out_list.extend([R_centers[0], R_centers[2]])
        return out_list

    def get_completed_colors(self):
        out_list = []
        faces = ['L', 'U', 'F', 'D', 'R', 'B']
        for face in faces:
            out_list.extend(Face(self, face).get_same_colors())
        return out_list

    def get_colors(self):
        out_list = []
        faces = ['L', 'U', 'F', 'D', 'R', 'B']
        for face in faces:
            out_list.extend(Face(self, face).get_colors())
        return out_list

    def copy(self):
        return Cube({c[1].copy() for c in self})

    def get_best_move(self, model, last_move, noise=None, return_cubes=False, return_moves=False):
        moves = ['F', 'U', 'R', 'D', 'L', 'B', 'F\'', 'U\'', 'R\'', 'D\'', 'L\'', 'B\'']
        
        if(last_move is not None):
            reverse_last_move = pc.Formula(last_move).reverse().__str__()
            moves.remove(reverse_last_move)
        
        predictions = []
        cubes = []
        for move in moves:
            new_cube = self.copy()
            new_cube(move)
            cubes.append(new_cube)
            predictions.append(model.predict_score(new_cube))

        if(noise):
            avg_prediction = int(np.floor(np.mean(predictions)))
            if(avg_prediction >= len(noise)): 
                sum_noise = noise[-1]
            else:
                sum_noise = noise[avg_prediction]
            predictions = [p + np.random.normal(0,sum_noise) for p in predictions]

        predictions_dict = dict(zip(moves, predictions))

        out_array = [moves[np.argmin(predictions)], predictions_dict]
        if(return_cubes): out_array.append(cubes)
        if(return_moves): out_array.append(moves)
        return out_array

    def solve(self, model, max_iter=100, show=True, noise=None):

        original_cube = self.copy()
        prediction_set = []
        last_move = None

        for _ in range(max_iter+1):
            if(self.is_completed()):
                break
            move, predictions = self.get_best_move(model, last_move, noise=noise)
            last_move = copy.copy(move)
            self = self(move)
            prediction_set.append(predictions)
            if(show): 
                print(move)
                print(self.__repr__())

        return original_cube, self, _, prediction_set

    def solve_search(self, model, max_iter=100, show=False):

        def get_new_sets(self, model, last_set):
            base_formula = last_set['Formula']
            
            if(base_formula.__str__() == '[]'): 
                last_move = last_set['Formula'][-1].__str__()
                base_formula = base_formula[0]
                last_move = None
            else: last_move = base_formula[-1]

            _, prediction_dict, cubes, moves = self.get_best_move(model, last_move, 
                return_cubes=True, return_moves=True)

            formulas = [pc.Formula(base_formula.__str__() + ' ' + move) for move in moves]
            predictions = [prediction_dict[move] for move in moves]

            out_sets = np.array(zip(formulas, cubes, predictions), dtype=last_set.dtype)
            return out_sets

        def prune_set(self, model, new_sets, alpha = 4):
            cube_prediction = model.predict_score(self)
            out_sets = new_sets[new_sets['Prediction'] < (cube_prediction + alpha)]
            return out_sets

        def get_minimum_state(saved_states):
            index = np.argmin(saved_states['Prediction'])
            out_state = saved_states[index]
            return out_state, index

        def stuck_in_loop(current_state):
            f = current_state['Formula']
            if(len(f) < 3): return False
            elif(f[-1] == f[-2] == f[-3]): return True
            else: return False

        original_cube = self.copy()
        saved_states = np.empty(0, dtype=[('Formula', object), ('Cube', object), ('Prediction', 'f8')])
        last_set = np.array([(pc.Formula(), self.copy(), 100)], dtype=saved_states.dtype)
        current_state = last_set

        for _ in range(max_iter+1):
            if(self.is_completed()): break
            if((_ > 0) & (saved_states.size == 0)): break
            new_sets = get_new_sets(self, model, last_set)
            pruned_set = prune_set(self, model, new_sets)
            if(pruned_set.size > 0): saved_states = np.append(saved_states, pruned_set)
            current_state, current_state_index = get_minimum_state(saved_states)
            saved_states = np.delete(saved_states, current_state_index)
            if(stuck_in_loop(current_state)): 
                current_state, current_state_index = get_minimum_state(saved_states)
                saved_states = np.delete(saved_states, current_state_index)
            self = current_state['Cube'].copy()
            last_set = current_state
            current_state

        return original_cube, self, _, current_state['Formula']

    def get_solution_prediction(self, solution, model):
        alt_cube = self.copy()
        predictions = []
        for move in solution:
            predictions.append(model.predict_score(alt_cube(move)))
        return predictions

class Game():

    def __init__(self, n_moves=100):

        def get_game_states(move_formula, solve_formula):
            cube = Cube()
            cube(move_formula)

            game_states = []
            for move in solve_formula:
                cube(move)
                game_state = cube.copy()
                game_states.append(game_state)

            return game_states

        def turn_2s_into_2_moves(formula):
            out_list = []
            for move in formula:
                str_move = str(move)
                if(("2" in str_move) & (len(str_move) == 2)):
                    out_move = str_move[0]
                    out_list.append(out_move)
                elif(("2" in str_move) & (len(str_move) == 3)):
                    out_move = str_move[0] + str_move[-1]
                    out_list.append(out_move)
                else:
                    out_move = str_move
                out_list.append(out_move)

            return pc.Formula(out_list)

        moves = ['F', 'U', 'R', 'D', 'L', 'B', 'F\'', 'U\'', 'R\'', 'D\'', 'L\'', 'B\'']
        move_list = np.random.choice(moves, n_moves)
        move_formula = pc.Formula(" ".join(move_list)).optimise()
        move_formula = turn_2s_into_2_moves(move_formula)
        self.solve_formula = copy.copy(move_formula)
        self.solve_formula.reverse()
        self.game_states = get_game_states(move_formula, self.solve_formula)

class Face():

    def __init__(self, cube, side, face_array=None):
        self.cube = cube
        self.side = side
        if(face_array is None):
            self.face = np.array(cube.get_face(side))
            relative_face_operators = self.get_relative_face_operators(side)
            self.relative_U, self.relative_R, self.relative_D, self.relative_L = \
                self.get_relative_faces(cube, relative_face_operators)
        else:
            self.face, self.relative_U, self.relative_R, self.relative_D, self.relative_L = \
                face_array

    def __getitem__(self, side):
        if(side == 'F'): return self.face
        elif(side == 'L'): return self.relative_L
        elif(side == 'R'): return self.relative_R
        elif(side == 'U'): return self.relative_U
        elif(side == 'D'): return self.relative_D

    def get_relative_face_operators(self, side):
        if(side == 'F'):   return ['r0U', 'r0R', 'r0D', 'r0L']
        elif(side == 'B'): return ['r2U', 'f1L', 'r2D', 'r0R']
        elif(side == 'D'): return ['r0F', 'r3R', 'r2B', 'r1L']
        elif(side == 'U'): return ['r2B', 'r1R', 'r0F', 'r3D']
        elif(side == 'L'): return ['r1U', 'r0F', 'r3D', 'f1B']
        elif(side == 'R'): return ['r3U', 'r0B', 'r1D', 'r0F']

    def get_relative_faces(self, cube, relative_face_operators):
        out_list = []
        for face_operator in relative_face_operators:
            operation, n_performances, side = face_operator
            face = np.array(cube.get_face(side))
            if(operation == 'r'): out_list.append(np.rot90(face, int(n_performances)))
            elif(operation== 'f'): out_list.append(np.fliplr(face))
        return out_list

    def rotate_face(self):
        F = np.rot90(self.face)
        U = np.rot90(self.relative_R)
        R = np.rot90(self.relative_D)
        D = np.rot90(self.relative_L)
        L = np.rot90(self.relative_U)
        face_list = [F, U, R, D, L]
        return Face(self.cube, self.side, face_list)

    def get_dummies(self, check_val='corners'):
        def check_corner(self):
            F_color = self['F'][0,0]
            F_center_color = self['F'][1,1]
            L_color = self['L'][0,2]
            L_center_color = self['L'][1,1]
            U_color = self['U'][2,0]
            U_center_color = self['U'][1,1]
            if((F_color == F_center_color) & (L_color == L_center_color) & (U_color == U_center_color)): 
                return 1
            else: 
                return 0
        
        def check_center(self):
            F_color = self['F'][0,1]
            F_center_color = self['F'][1,1]
            U_color = self['U'][2,1]
            U_center_color = self['U'][1,1]
            if((F_color == F_center_color) & (U_color == U_center_color)): 
                return 1
            else: 
                return 0

        if(check_val == 'corners'): check_func = check_corner
        elif(check_val == 'centers'): check_func = check_center

        out_vals = []
        rotated_face = self
        for _ in range(4):
            val = check_func(rotated_face)
            rotated_face = rotated_face.rotate_face()
            out_vals.append(val)
        return out_vals

    def get_same_colors(self):
        def check_corner(face):
            F_color = face['F'][0,0]
            F_center_color = face['F'][1,1]
            if((F_color == F_center_color)): return 1
            else: return 0

        def check_center(face):
            F_color = face['F'][0,1]
            F_center_color = face['F'][1,1]
            if((F_color == F_center_color)): return 1
            else: return 0

        out_vals = []
        rotated_face = self
        for _ in range(4):
            corner = check_corner(rotated_face)
            center = check_center(rotated_face)
            out_vals.append(corner)
            out_vals.append(center)
            rotated_face = rotated_face.rotate_face()
        return out_vals

    def get_colors(self):
        def color_to_number(color):
            out_array = [0,0,0,0,0,0]
            if(color.__str__() == '[g]'):   ind = 0
            elif(color.__str__() == '[y]'): ind = 1
            elif(color.__str__() == '[o]'): ind = 2
            elif(color.__str__() == '[w]'): ind = 3
            elif(color.__str__() == '[r]'): ind = 4
            elif(color.__str__() == '[b]'): ind = 5
            out_array[ind] = 1
            return out_array

        def get_vals(face):
            corner_color = color_to_number(face['F'][0,0])
            center_color = color_to_number(face['F'][0,1])
            return corner_color, center_color

        out_vals = []
        rotated_face = self
        for _ in range(4):
            corner, center = get_vals(rotated_face)
            out_vals.extend(corner)
            out_vals.extend(center)
            rotated_face = rotated_face.rotate_face()
        return out_vals


    def flatten_face(self):
        return [item for sublist in self.face for item in sublist]

    def is_completed(self):
        flattened = self.flatten_face()
        return all([item == flattened[0] for item in flattened])

    def get_percent_completed(self):
        flattened = self.flatten_face()
        center = flattened[4]
        percent_completed = sum([item == center for item in flattened]) / 9.0
        return percent_completed

class Model():

    def __init__(self):
        self.training_data = None
        self.regression_model = None
                
    def create_training_data(self, n_games, n_moves, show_progress = True):
        def create_pd_dataframe(reg_list, n_vals=48):
            col_names = ['moves_left']
            for val in range(n_vals): col_names.append(str(val))
            out_df = pd.DataFrame(reg_list, columns = col_names, dtype = np.int8)
            for val in range(n_vals): out_df[str(val)] = out_df[str(val)].astype(np.bool)
            reg_list = []
            return out_df, reg_list

        def append_to_df(training_data, reg_list):
            dt = [(col, training_data.dtypes[col].descr[0][1]) for col in training_data.columns]
            append_array = np.array(reg_list, dtype=dt)
            append_data = pd.DataFrame(append_array)
            training_data = training_data.append(append_data, ignore_index=True)
            reg_list = []
            return training_data, reg_list

        if(show_progress):
            display_iterations = int(n_games / 50)
            if(display_iterations == 0): display_iterations = 1

        reg_list = []
        for _ in range(n_games):
            
            g = Game(n_moves)

            for (index, c) in enumerate(g.game_states):
                feature_list = c.get_colors()
                moves_away = [len(g.game_states) - index]
                reg_list_entry = []
                reg_list_entry.extend(moves_away)
                reg_list_entry.extend(feature_list)
                reg_list.append(tuple(reg_list_entry))
                if((_==0) & (index==0)): training_data, reg_list = create_pd_dataframe(reg_list, len(reg_list[0])-1)

            if(_ % 5000 == 0): training_data, reg_list = append_to_df(training_data, reg_list)
            if(show_progress & ((_ % display_iterations) == 0)): print("Finished with " + str(_) + " games")

        if(reg_list != []): training_data, reg_list = append_to_df(training_data, reg_list)
        self.training_data = training_data

    def train_model(self, hidden_layer_sizes  = (100,)):
        if(self.training_data is None):
            return None

        y = self.training_data.ix[:,0]
        x = self.training_data.ix[:,1:]

        reg = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes)
        reg.fit(x,y)

        self.regression_model = reg

    def predict_score(self, cube):
        x_vals = np.array(cube.get_colors()).reshape(1,-1)
        return self.regression_model.predict(x_vals)[0]

class Test():

    def __init__(self, model):
        self.model = model

    def correct_move_selection(self, n_moves=10):       
        best_move_average_rank = []
        for move_index in range(1,n_moves):
            best_moves = []
            for _ in range(100):
                c = Cube()
                scramble_moves, solution = c.scramble_cube(move_index)
                best_move = solution.pop(0).__repr__()
                best_guess, prediction_dict = c.get_best_move(self.model, None)

                best_move_score = prediction_dict[best_move]
                prediction_list = prediction_dict.values()
                prediction_list.sort()
                best_move_rank = prediction_list.index(best_move_score)+1

                best_moves.append(best_move_rank)

            best_move_average_rank.append(np.average(best_moves))
        return best_move_average_rank

    def correct_cube_predictions(self, n_moves=10):
        cube_predictions_average = []
        for move_index in range(n_moves):
            cube_predictions = []
            for _ in range(100):
                c = Cube()
                scramble_moves, solution = c.scramble_cube(move_index)
                cube_predictions.append(self.model.predict_score(c))
            cube_predictions_average.append(np.average(cube_predictions))
        return cube_predictions_average

    def correct_cubes_solved(self, n_moves=10, max_iter=100, solve_search=False):

        cubes_solved_average = []
        for move_index in range(n_moves):
            cubes_solved = []
            for _ in range(100):
                c = Cube()
                
                if(solve_search): search_func=c.solve_search
                else: search_func=c.solve
        

                scramble_moves, solution = c.scramble_cube(move_index)
                orig_cube, final_cube, iterations, pred_set = search_func(self.model, max_iter=max_iter, show=False)
                if(iterations == max_iter): solve_val = 0.0
                else: solve_val = 1.0
                cubes_solved.append(solve_val)
            cubes_solved_average.append(np.average(cubes_solved))
        return cubes_solved_average

    def all_diagnostics(self, n_moves=10):
        t_results_1 = self.correct_move_selection(n_moves)
        t_results_2 = self.correct_cube_predictions(n_moves)
        t_results_3 = self.correct_cubes_solved(n_moves)
        return t_results_1, t_results_2, t_results_3