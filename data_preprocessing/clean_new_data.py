import pandas as pd
from expand_text import expand_text
import re

def remove_tags(sent):
    # return sent.replace(r'<sub alias="([^"]*)">[^<]*</sub>', r'\1', regex=True)
    return re.sub(r'<sub alias="([^"]*)">[^<]*</sub>', r'\1', sent)


if __name__ == "__main__":
    new_data_raw = "./new_questions/V3_Copy_of_data2_upv_changes_raw.xlsx"

    df = pd.read_excel(new_data_raw)

    q_json = []
    a_json = []
    added_classes = set()

    for index, row in df.iterrows():
        expanded = expand_text(row["Question"]) # expand (a|b|c)

        for q in expanded:
            q_without_number = re.sub(r'^\d+\.', '', q, count=1)
            q_json.append({"question": q_without_number, "class": row["Class"]})
            # q_json.append({"question": q, "class": row["Class"]})

        if int(row["Class"]) not in added_classes:
            a_json.append({"answer": remove_tags(row["Answer"]), "class": row["Class"]})
            added_classes |= {int(row["Class"])}

    q_data = pd.DataFrame.from_records(q_json)
    a_data = pd.DataFrame.from_records(a_json)

    q_data.to_excel("./new_questions/M_Q76_questions_V3.xlsx")
    a_data.to_excel("./new_questions/M_Q76_answers_V3.xlsx")
