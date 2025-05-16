from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatOpenAI()

#schema
class Review(TypedDict):
    summary:Annotated[str, "A brief summary of the review"]
    sentiment:Annotated[str, "Return the sentiment of the review either Negative, Postive, or Neutral"]
    # sentiment:Annotated[Literal["pos", "neg"], "Return the sentiment of the review either Negative, Postive, or Neutral"]

sturctured_output = model.with_structured_output(Review)

result = sturctured_output.invoke('''I recently tried the new electric SUV from Tesla, and I have mixed feelings about it. The acceleration and handling were absolutely top-notch — it feels like driving the future. However, the build quality on the interior was underwhelming for the price point. There were some rattling noises, and the infotainment system froze once during my test drive. Overall, it's a great step forward for EVs, but there's room for improvement.''')

print(result)

review = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Nitish Singh
"""

class AnotherReview(Review):
   key_themes:Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
   pros:Annotated[Optional[list[str]], "Write down all the pros inside a list"]
   cons:Annotated[Optional[list[str]], "Write down all the cons inside a list"]

sturctured_output2 = model.with_structured_output(AnotherReview)

result = sturctured_output2.invoke(review)
print(result)