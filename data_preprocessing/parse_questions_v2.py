import pandas as pd
from expand_text import expand_text
import re

if __name__ == "__main__":
    df = pd.read_excel("../data/V4_alternative_questions_UPV.xlsx", usecols=[1, 2, 3])
    df['question'] = df['question'].str.replace(r'^\d+\.\s', '', regex=True)
    df["answer"] = df["answer"].str.replace(r'<sub alias="([^"]*)">[^<]*</sub>', r'\1', regex=True)
    print(df['class'].nunique())
    df.loc[df['class'] == 78, 'class'] = 77
    df.loc[df['class'] > 67, 'class'] -= 1
    df.loc[df['class'] > 60, 'class'] -= 1
    print(df['class'].nunique())
    print(df.head())
    df.to_excel("FAQv4_raw2.xlsx")

    q_json = []
    a_json = []
    qa_json = []

    for i in range (0, df.shape[0]):
        q = df['question'][i]
        expanded = expand_text(q)
        print(len(expanded))
        questions = [q.replace('  ', ' ').strip() for q in expanded]
        print(questions)
        answer = df["answer"][i]
        cls = df["class"][i]
        for q in questions:
            q_json.append({"question": q, "class": cls})
            qa_json.append({"question": q, "answer": answer})
        a_json.append({"answer": answer, "class": cls})

    q_data = pd.DataFrame.from_records(q_json)
    a_data = pd.DataFrame.from_records(a_json)
    qa_data = pd.DataFrame.from_records(qa_json)

    q_data.to_excel("FAQv4_questions.xlsx")
    a_data.to_excel("FAQv4_answers.xlsx")
    q_data.to_csv("FAQv4_questions.csv", sep="\t")
    a_data.to_csv("FAQv4_answers.csv", sep="\t")

    qa_data.to_csv("FAQv4_QA.csv", sep="\t")
