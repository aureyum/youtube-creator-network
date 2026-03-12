import pandas as pd

df = pd.read_csv(r"C:\Users\Aurelie\Downloads\archive\CAvideos.csv")

df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
print(len(df))

# keep earliest trending date per video
df = df.sort_values('trending_date')
df = df.drop_duplicates(subset='video_id', keep='first')

print(len(df))