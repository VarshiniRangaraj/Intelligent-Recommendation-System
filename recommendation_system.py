import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("products.csv")

data.fillna("", inplace=True)

data["combined_features"] = data["category"] + " " + data["description"]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data["combined_features"])

cosine_sim = cosine_similarity(tfidf_matrix)


def recommend(product_name):
    product_name = product_name.lower().strip()

    matches = data[data["product_name"].str.lower().str.contains(product_name)]

    if matches.empty:
        print("Product not found. Try a different name.")
        return

    product_index = matches.index[0]

    similarity_scores = list(enumerate(cosine_sim[product_index]))

    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    print("\nRecommended Products For You:\n")

    count = 0
    for i in sorted_scores[1:]:
        product = data.iloc[i[0]]["product_name"]
        score = i[1]

        if score > 0.1:
            print(f"{product} ")
            count += 1

        if count == 5:
            break

    if count == 0:
        print("No strong recommendations found. Try another product.")


print("INTELLIGENT E-COMMERCE RECOMMENDATION SYSTEM")

print("\nAvailable Products:\n")

for i, row in data.iterrows():
    print(f"{i+1}. {row['product_name']}")

print("\n")

choice = input("Select a product to BUY: ")

print(f"\nYou selected: {choice}")

recommend(choice)