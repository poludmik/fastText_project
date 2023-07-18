import pickle
import pandas as pd
import re


def convert_pkl_Q50_to_xlsx(path_to_pkl: str): 
    """
    Assumes that each question string has a number in the beggining. The number is removed with regex.
    """
    with open(path_to_pkl, 'rb') as f:
        data = pickle.load(f)
    
    questions = {"question": [], "class": []}
    answers = {"answer": [], "class":[]}
    for idx, ans in enumerate(data):
        for q in data[ans]['questions']:
            if len(q) > 3:
                q_without_number = re.sub("^[0-9]+\. ", "", q, count=1)
                questions['question'].append(q_without_number)
                questions['class'].append(idx)
        answers['answer'].append(ans)
        answers['class'].append(idx)

    questions = pd.DataFrame(questions)
    path = '.'.join(path_to_pkl.split(".")[:-1]) + '.xlsx'
    questions.to_excel(path, sheet_name='sheet1', index=True)

    answers = pd.DataFrame(answers)
    path = '.'.join(path_to_pkl.split(".")[:-1]) + '_answers.xlsx'
    answers.to_excel(path, sheet_name='sheet1', index=True)



if __name__ == "__main__":
    path_to_pkl_Q50 = '../upv_search/expanded_data_all2.pkl'
    
    convert_pkl_Q50_to_xlsx(path_to_pkl_Q50)
