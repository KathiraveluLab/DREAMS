# Thematic Analysis Module
This module performs thematic analysis of recovery project data using **Google Gemini**. Implemented at - **utils/llms.py**

It takes clusters of positive and negative keywords from participants’ entries and generates high-level themes with warm, empathetic explanations.

The results are stored in MongoDB and can be accessed via the project dashboard.

## How It Works
- Input:
        - **user_id** Participant identifier
        - **positive_keywords** List of Keywords from Positive Experience
        - **negative_keywords**  List of ketwords from Negative Experiecne

- Processing:
        - The module sends the keywords to **Gemini** with a qualitative sociology prompt.
        - Gemini generates 2–3 overarching themes for each cluster with short, empathetic meanings.
        - The response is parsed, validated, and stored in the thematic_analysis collection in MongoDB.
- Output:
        -  JSON object with **positive** and **negative** theme clusters.

### Example Input

```json
generate(
  user_id="101",
  positive_keywords=["family", "support", "hope", "progress"],
  negative_keywords=["isolation", "fear", "relapse"]
)
```
### Example Output
```json
{
  "positive": [
    {
      "theme": "Family Support",
      "meaning": "The participant feels uplifted by their family’s encouragement."
    },
    {
      "theme": "Personal Growth",
      "meaning": "They recognize their progress and feel hopeful about recovery."
    }
  ],
  "negative": [
    {
      "theme": "Isolation",
      "meaning": "The participant struggles with loneliness during difficult times."
    },
    {
      "theme": "Fear of Relapse",
      "meaning": "They feel anxious about falling back into old patterns."
    }
  ]
}
```



