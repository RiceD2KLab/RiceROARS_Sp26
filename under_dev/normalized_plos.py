import pandas as pd
import ast

df = pd.read_csv("rice_ga_credential_plos.csv")

# drop cells with zero PLOs
df = df[df["plo_count"] > 0]

rows = []

for _, r in df.iterrows():
    plo_list = r["plos_list"]

    if isinstance(plo_list, str):
        plo_list = ast.literal_eval(plo_list)

    for outcome in plo_list:
        rows.append({
            "department_name": r["department_name"],
            "credential_title": r["credential_title"],
            "level": r["level"],
            "outcome_text": outcome,
            "credential_url": r["credential_url"],
        })

plos_df = pd.DataFrame(rows)
plos_df.to_csv("PLOs_long.csv", index=False)


