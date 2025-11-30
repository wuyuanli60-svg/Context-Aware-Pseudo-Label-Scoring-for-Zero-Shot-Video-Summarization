import numpy as np
import csv

import numpy as np

def evaluate_summary_fscore(predicted_summary, user_summary, eval_method):
    """
    返回 (f_score_percent, precision_percent, recall_percent)
    - eval_method == 'avg' (TVSum): 返回所有用户的 mean P/R/F1
    - eval_method == 'max' (SumMe): 返回 F1 最大用户对应的 P/R/F1
    """
    max_len = max(len(predicted_summary), user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_list, p_list, r_list = [], [], []

    for u in range(user_summary.shape[0]):
        G = np.zeros(max_len, dtype=int)
        G[:user_summary.shape[1]] = user_summary[u]

        inter = int((S & G).sum())
        s_sum = int(S.sum())
        g_sum = int(G.sum())

        p = 0.0 if s_sum == 0 else inter / s_sum
        r = 0.0 if g_sum == 0 else inter / g_sum
        f = 0.0 if (p + r) == 0 else (2.0 * p * r) / (p + r)

        p_list.append(p)
        r_list.append(r)
        f_list.append(f)

    if eval_method == 'max':  # SumMe
        idx = int(np.argmax(f_list))
        return f_list[idx] * 100.0, p_list[idx] * 100.0, r_list[idx] * 100.0
    else:  # 'avg' for TVSum
        return (np.mean(f_list) * 100.0,
                np.mean(p_list) * 100.0,
                np.mean(r_list) * 100.0)




def evaluate_summary_fscore_all(predicted_summary, user_summary, eval_method):
    ''' 
    Function that evaluates the predicted summary using F-Score. 

    Inputs:
        predicted_summary: numpy (binary) array of shape (n_frames)
        user_summary: numpy (binary) array of shape (n_users, n_frames)
        eval_method: method for combining the F-Scores for each comparison with user summaries - values: 'avg' or 'max'
    Outputs:
        max (TVSum) or average (SumMe) F-Score between users 

    '''
    max_len = max(len(predicted_summary),user_summary.shape[1])
    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary
    results = []
    f_scores = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G
        
        # Compute precision, recall, f-score
        precision = 0 if sum(S) == 0 else sum(overlapped)/sum(S)
        recall = 0 if sum(G) == 0 else sum(overlapped)/sum(G)
        if (precision+recall==0):
            f_scores.append(0)
            results.append((0,0,0))
        else:
            f_scores.append(2*precision*recall*100/(precision+recall))
            results.append(
                [
                    precision * 100,
                    recall * 100,
                    (2 * precision * recall * 100 / (precision + recall)),
                ]
            )
    results = np.array(results)
    
    if eval_method == "max":
        x = [y[2] for y in results]
        max_values = np.argmax(x)
        return results[max_values]
    else:
        return np.mean(results, axis=0)  
