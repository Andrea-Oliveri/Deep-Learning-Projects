import pickle
import numpy as np
import pandas

filename = "../measures/MLP_measures.pkl"

with open(filename, "rb") as file:
    data = pickle.load(file)
    

for i in range(len(data)):
    params, res = data[i]
    
    if type(res) == dict:
        new_res = []
        
        for k in ['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']:
            new_res.append( res[k] )
        
        
        data[i] = (params, new_res)
    

if 'Aux' not in filename:
    data = [m for m in data if 'beta' not in m[0]] # Corrects an error in ConvNet_measures. MLP_measures untouched by this.
    # results[1] for test_loss, results[-1] for test_acc
    data_final_test_acc = [{**dic, 'test_acc': results[-1][np.argmin(results[1])].item(), 'n_epochs_run': len(results[-1])} for dic, results in data]
else:
    # results[1][:,-1] for tot_test_loss, results[-1][:-1] for final_test_acc
    data_final_test_acc = [{**dic, 'test_acc': results[-1][np.argmin(results[1][:,-1])][-1].item(), 'n_epochs_run': len(results[-1])} for dic, results in data]


df = pandas.DataFrame(data_final_test_acc).sort_values('test_acc', ascending = False)


print(df[:20].to_string(max_rows = None, max_cols = None, index = False))