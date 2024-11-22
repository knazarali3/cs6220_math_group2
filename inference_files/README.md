# **Instructions for llemma_sat_math_answers.txt**

**The file lines are set up in the following format:** {'Iteration': integer, 'Question Id': string of question's id, 'Generated Answer': string of model's generated output} 


**To extract information from the file, use the following code:**

    file = open("llemma_math_sat_answers.txt", "r")
    for i in range(6290):
      line = eval(file.readline())
      iteration = line["Iteration"]
      question_id = line["Question Id"]
      generated_answer = line["Generated Answer"]

# **Instructions for llemma_competition_math_answers.txt**
**The file lines are set up in the following format:** {'Iteration': integer, 'Question': string of question, 'Type': string of question's type, 'Level': string of question's level, 'Generated Answer': string of model's generated output} 


**To extract information from the file, use the following code:**

    file = open("llemma_competition_sat_answers.txt", "r")
    for i in range(5000):
      line = eval(file.readline())
      iteration = line["Iteration"]
      question = line["Question"]
      type = line["Type"]
      level = ["Level"]
      generated_answer = line["Generated Answer"]
