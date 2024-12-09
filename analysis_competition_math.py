
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def format_llema_inference_competition_math(llema_file_path, test_dataset_df):
    with open(llema_file_path, "r") as file:
        data = [eval(line.strip()) for line in file]
    df = pd.DataFrame(data)
    llema_test_dataset= df.merge(test_dataset_df, left_on='Iteration', right_on='row_index')
    llema_test_dataset["level:type"] = llema_test_dataset["level"] + ": " + llema_test_dataset["type"]

    llema_test_dataset["llema_is_correct"] =  (llema_test_dataset["Final Solution"] == llema_test_dataset["extracted_solution"])

    llema_test_dataset = llema_test_dataset.rename(columns={"Final Solution": "llema_solution"})

    llema_3_results = llema_test_dataset[["row_index", "level", "type", "llema_is_correct", "llema_solution"]]
    return llema_3_results

def format_mistral_inference_competition_math(mistral_file_path, test_dataset_df):
    df = pd.read_csv(mistral_file_path)
    mistral_test_dataset= df.merge(test_dataset_df, left_on='row_index', right_on='row_index')
    mistral_test_dataset["mistral_is_correct"] = (mistral_test_dataset["Run_1"] == mistral_test_dataset["extracted_solution"])
    mistral_test_dataset["level:type"] = mistral_test_dataset["level"] + ": " + mistral_test_dataset["type"]

    mistral_test_dataset = mistral_test_dataset.rename(columns={"Run_1": "mistral_solution"})

    mistral_results = mistral_test_dataset[["row_index", "level", "type", "mistral_is_correct", "mistral_solution"]]
    return mistral_results    

def format_llama_inference_competition_math(llama_file_path, test_dataset_df):
    df = pd.read_csv(llama_file_path)
    llama3_test_dataset= df.merge(test_dataset_df, left_on='row_index', right_on='row_index')
    llama3_test_dataset["llama3_is_correct"] = (llama3_test_dataset["Run_1"] == llama3_test_dataset["extracted_solution"])
    llama3_test_dataset["level:type"] = llama3_test_dataset["level"] + ": " + llama3_test_dataset["type"]

    llama3_test_dataset = llama3_test_dataset.rename(columns={"Run_1": "llama_solution"})

    llama3_results = llama3_test_dataset[["row_index", "level", "type", "llama3_is_correct", "llama_solution"]]
    return llama3_results

def format_qwen_inference_competition_math(qwen_file_path, test_dataset_df):
    df = pd.read_csv(qwen_file_path)

    qwen_test_dataset= df.merge(test_dataset_df, left_on='row_index', right_on='row_index')
    qwen_test_dataset["qwen_is_correct"] = (qwen_test_dataset["Run_1"] == qwen_test_dataset["extracted_solution"])
    qwen_test_dataset["level:type"] = qwen_test_dataset["level"] + ": " + qwen_test_dataset["type"]

    qwen_test_dataset = qwen_test_dataset.rename(columns={"Run_1": "qwen_solution"})

    qwen_results = qwen_test_dataset[["row_index", "level", "type", "level:type", "qwen_is_correct", "qwen_solution"]]
    return qwen_results

def give_all_results(llama_file_path, llema_file_path, qwen_file_path, mistral_file_path, test_dataset_df):
    llama3_results = format_llama_inference_competition_math(llama_file_path, test_dataset_df)
    llema_results = format_llema_inference_competition_math(llema_file_path, test_dataset_df)
    qwen_results = format_qwen_inference_competition_math(qwen_file_path, test_dataset_df)
    mistral_results = format_mistral_inference_competition_math(mistral_file_path, test_dataset_df)

    # Check where all the answers correct column is True
    # i.e. all models got the answer correct
    all_results = ((llama3_results.merge(llema_results)).merge(qwen_results)).merge(mistral_results)
    all_results["all_correct"] = (all_results["llama3_is_correct"] == True) & (all_results["llema_is_correct"] == True) & (all_results["qwen_is_correct"] == True) & (all_results["mistral_is_correct"]==True)

    all_results["all_incorrect"] = (all_results["llama3_is_correct"] == False) & (all_results["llema_is_correct"] == False) & (all_results["qwen_is_correct"] == False) & (all_results["mistral_is_correct"]==False)

    all_results["at_least_3_correct"] = (
        all_results[["llama3_is_correct", "llema_is_correct", "qwen_is_correct", "mistral_is_correct"]]
        .sum(axis=1) >= 3
    )
    all_results["at_least_2_correct"] = (
        all_results[["llama3_is_correct", "llema_is_correct", "qwen_is_correct", "mistral_is_correct"]]
        .sum(axis=1) >= 2
    )

    all_results["at_least_1_correct"] = (
        all_results[["llama3_is_correct", "llema_is_correct", "qwen_is_correct", "mistral_is_correct"]]
        .sum(axis=1) >= 1
    )

    # Check wehere all answers solution column have the same answer
    # i.e. all models got the same answer
    all_results["all_agree"] = (
        (all_results["llema_solution"] == all_results["llama_solution"]) &
        (all_results["qwen_solution"] == all_results["llama_solution"]) &
        (all_results["mistral_solution"] == all_results["llama_solution"]) &
        (all_results["llema_solution"] != "Error Parsing Response")
    )
    return all_results

def plot_bigram_combination_accuracies(all_results):
    #Llama + Mistral, Llama + Qwen, Llama + Llema, Mistral + Qwen, Mistral + Llema, Qwen + Llema
    llamaplusmistral = all_results[(all_results["llama3_is_correct"] == True) & (all_results["mistral_is_correct"] == True)]
    llamaplusqwen = all_results[(all_results["llama3_is_correct"] == True) & (all_results["qwen_is_correct"] == True)]
    llamaplusllema = all_results[(all_results["llama3_is_correct"] == True) & (all_results["llema_is_correct"] == True)]
    mistralplusqwen = all_results[(all_results["mistral_is_correct"] == True) & (all_results["qwen_is_correct"] == True)]
    mistralplusllema = all_results[(all_results["mistral_is_correct"] == True) & (all_results["llema_is_correct"] == True)]
    qwenpluslemma = all_results[(all_results["qwen_is_correct"] == True) & (all_results["llema_is_correct"] == True)]

    accuracies = pd.DataFrame(list(zip([len(llamaplusmistral), len(llamaplusqwen), len(llamaplusllema), len(mistralplusqwen), len(qwenpluslemma), len(mistralplusllema)])))
    total_sum = len(all_results)
    accuracies.index = ["llama+mistral", "llama+qwen", "llama+llema", "mistral+qwen", "mistral+llema", "qwen+lemma"]
    accuracies.columns = ["percent"]
    accuracies["percent"] = (accuracies["percent"] / total_sum)*100
    accuracies = accuracies.sort_values(by=['percent'], ascending=False)

    fig, ax = plt.subplots(figsize=(12, 12))  
    bars = plt.bar(accuracies.index, accuracies["percent"], color = "lightgreen")
    ax.bar_label(bars, fmt="%.1f%%", label_type='edge', fontsize=25) 
    ax.set_xlabel("Percent Correct By 2 Model Combination", labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=15, width=2.5, length=10)

    ax.set_xlabel('Percent Correct By 2 Model Combination', fontsize=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return plt
    #plt.show()

def plot_trigram_combination_accuracies(all_results):
    llama_mistral_qwen = all_results[
        (all_results["llama3_is_correct"] == True) & 
        (all_results["mistral_is_correct"] == True) & 
        (all_results["qwen_is_correct"] == True)
    ]

    llama_mistral_llema = all_results[
        (all_results["llama3_is_correct"] == True) & 
        (all_results["mistral_is_correct"] == True) & 
        (all_results["llema_is_correct"] == True)
    ]

    llama_qwen_llema = all_results[
        (all_results["llama3_is_correct"] == True) & 
        (all_results["qwen_is_correct"] == True) & 
        (all_results["llema_is_correct"] == True)
    ]

    mistral_qwen_llema = all_results[
        (all_results["mistral_is_correct"] == True) & 
        (all_results["qwen_is_correct"] == True) & 
        (all_results["llema_is_correct"] == True)
    ]

    # Calculate accuracies
    accuracies_trigram = pd.DataFrame(
        list(
            zip(
                [len(llama_mistral_qwen), len(llama_mistral_llema), len(llama_qwen_llema), len(mistral_qwen_llema)],
                ["llama+mistral+qwen", "llama+mistral+llema", "llama+qwen+llema", "mistral+qwen+llema"]
            )
        ),
        columns=["correct_count", "combination"]
    )

    total_sum = len(all_results)
    accuracies_trigram["percent"] = (accuracies_trigram["correct_count"] / total_sum) * 100
    accuracies_trigram = accuracies_trigram.sort_values(by=['percent'], ascending=False)

    fig, ax = plt.subplots(figsize=(12, 12))
    bars = plt.bar(accuracies_trigram["combination"], accuracies_trigram["percent"], color="skyblue")

    ax.bar_label(bars, fmt="%.1f%%", label_type='edge', fontsize=20)
    ax.set_title("Percent Correct By 3 Model Combination", fontsize=20, pad=20)
    ax.set_xlabel("Model Combinations", fontsize=16)
    ax.set_ylabel("Accuracy Percentage", fontsize=16)

    ax.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)
    plt.xticks(rotation=45, ha="right")

    #plt.tight_layout()
    return plt
    #plt.show()

def plot_percent_correct_each_model(all_results):
    llama = all_results[all_results["llama3_is_correct"] == True]
    llema = all_results[all_results["llema_is_correct"] == True]
    mistral = all_results[all_results["mistral_is_correct"] == True]
    qwen = all_results[all_results["qwen_is_correct"] == True]

    accuracies = pd.DataFrame(list(zip([len(llama), len(llema), len(mistral), len(qwen)])))
    total_sum = len(all_results)
    accuracies.index = ["llama", "llema", "mistral", "qwen"]
    accuracies.columns = ["percent"]
    accuracies["percent"] = (accuracies["percent"] / total_sum)*100
    accuracies = accuracies.sort_values(by=['percent'], ascending=False)

    fig, ax = plt.subplots(figsize=(12, 12))  
    bars = plt.bar(accuracies.index, accuracies["percent"], color = "lightgreen")
    ax.bar_label(bars, fmt="%.1f%%", label_type='edge', fontsize=25) 
    ax.set_xlabel("Percent Correct By Each Model", labelpad=20)

    ax.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)

    ax.set_xlabel('Perent Correct By Each Model', fontsize=20)

def plot_percent_correct_excluding_all_correct_all_wrong(all_results):
    mixed_llama = all_results[(all_results["llama3_is_correct"] == True) & (all_results["all_correct"] == False) & (all_results["all_incorrect"]==False)]
    mixed_llema = all_results[(all_results["llema_is_correct"] == True) & (all_results["all_correct"] == False) & (all_results["all_incorrect"]==False)]
    mixed_mistral = all_results[(all_results["mistral_is_correct"] == True) & (all_results["all_correct"] == False) & (all_results["all_incorrect"]==False)]
    mixed_qwen = all_results[(all_results["qwen_is_correct"] == True) & (all_results["all_correct"] == False) & (all_results["all_incorrect"]==False)]

    mixed_accuracies = pd.DataFrame(list(zip([len(mixed_llama), len(mixed_llema), len(mixed_mistral), len(mixed_qwen)])))
    total_sum = len(all_results[(all_results["all_correct"] == False) & (all_results["all_incorrect"]==False)])
    mixed_accuracies.index = ["llama", "llema", "mistral", "qwen"]
    mixed_accuracies.columns = ["percent"]
    mixed_accuracies["percent"] = (mixed_accuracies["percent"] / total_sum)*100
    mixed_accuracies = mixed_accuracies.sort_values(by=["percent"], ascending=False)

    fig, ax = plt.subplots(figsize=(12, 12))  

    bars = plt.bar(mixed_accuracies.index, mixed_accuracies["percent"], color="lightgreen")


    ax.bar_label(bars, fmt="%.1f%%", label_type='edge', fontsize=25) 

    ax.set_xlabel("Percent Correct By Each Model", labelpad=20)

    ax.tick_params(axis='both', which='major', labelsize=20, width=2.5, length=10)

    ax.set_xlabel('Perent Correct By Each Model', fontsize=20)

    ax.set_xlabel("Percent Correct Excluding all correct and all incorrect", labelpad=10)

def give_ensemble_model_atleast1_piechart(all_results):

    ensemble_model_atleast1_correct = pd.DataFrame( all_results['at_least_1_correct'].value_counts() )

    labels = ['Other', 'At least 1 Model Correct']
    sizes = [ ensemble_model_atleast1_correct['count'].iloc[0],ensemble_model_atleast1_correct['count'].iloc[1]]
    colors = ['#ff9999', '#66b3ff']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.axis('equal')
    plt.legend()
    plt.title(' Questions That At Least 1 Model Gave Correct Answers')

    return plt

def give_ensemble_model_atleast2_piechart(all_results):

    ensemble_model_atleast2_correct = pd.DataFrame( all_results['at_least_2_correct'].value_counts() )

    labels = ['Other', 'At least 2 Model Correct']
    sizes = [ ensemble_model_atleast2_correct['count'].iloc[0],ensemble_model_atleast2_correct['count'].iloc[1]]
    colors = ['#ff9999', '#66b3ff']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.axis('equal')
    plt.legend()
    plt.title(' Questions That At Least 2 Models Gave Correct Answers')

    return plt

def give_ensemble_model_atleast3_piechart(all_results):
    ensemble_model_atleast3_correct = pd.DataFrame( all_results['at_least_3_correct'].value_counts() )

    labels = ['Other', 'At least 3 Model Correct']
    sizes = [ ensemble_model_atleast3_correct['count'].iloc[0],ensemble_model_atleast3_correct['count'].iloc[1]]
    colors = ['#ff9999', '#66b3ff']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.axis('equal')
    plt.legend()
    plt.title(' Questions That At Least 3 Models Gave Correct Answers')

    return plt

def give_ensemble_model_all_incorrect_piechart(all_results):

    ensemble_model_all_incorrect = pd.DataFrame( all_results['all_incorrect'].value_counts() )

    labels = ['Other', 'All Incorrect']
    sizes = [ ensemble_model_all_incorrect['count'].iloc[0],ensemble_model_all_incorrect['count'].iloc[1]]
    colors = ['lightgreen', 'red']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.axis('equal')
    plt.legend()
    plt.title(' Questions That All Models Gave Incorrect Answers')

    return plt

def give_ensemble_model_all_correct_piechart(all_results):

    ensemble_model_all_correct = pd.DataFrame( all_results['all_correct'].value_counts() )

    labels = ['Other', 'All Correct']
    sizes = [ ensemble_model_all_correct['count'].iloc[0],ensemble_model_all_correct['count'].iloc[1]]
    colors = ['red', 'lightgreen']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.axis('equal')
    plt.legend()
    plt.title(' Questions That All Models Gave Correct Answers')

    return plt

def give_ensemble_model_all_agree_piechart(all_results):

    ensemble_model_all_agree = pd.DataFrame(all_results["all_agree"].value_counts())

    labels = ['Agree', 'Disagree']
    sizes = [ ensemble_model_all_agree['count'].iloc[0],ensemble_model_all_agree['count'].iloc[1]]
    colors = ['#ff9999', '#66b3ff']

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.axis('equal')
    plt.legend()
    plt.title(' Questions That All Models Agreed on vs Disagreed on Answer')

    return plt

def get_competition_math_level_type_distribution(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]

    percent_each_leveltype = percent_each_leveltype.sort_values(by=['Level:Type'], ascending=False)
    percent_each_leveltype

    fig, ax = plt.subplots(figsize=(12, 7))  

    bars = plt.barh(percent_each_leveltype.index, percent_each_leveltype["Percent of Level:Type in Dataset"], color = "skyblue")

    ax.bar_label(bars, fmt="%.1f%%", label_type='edge', fontsize=10) 

    ax.set_xlabel("Percent of Level:Type", labelpad=10)

    return plt

def get_competition_math_individual_breakdown_all_correct(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]
    
    level_types = all_results['level:type'].unique()

    all_level_types = []
    questions_right = []
    questions_wrong = []

    percent_questions_right_total = []
    percent_questions_wrong_total = []

    percent_questions_right_total_raw = []

    for cat in level_types:
        right_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["all_correct"] == True)) ]["row_index"]
        right_string_row_idxquestions = ", ".join(map(str, right_row_indices_of_questions))

        wrong_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["all_correct"] == False)) ]["row_index"]
        wrong_string_row_idxquestions = ", ".join(map(str, wrong_row_indices_of_questions))
        
        all_level_types.append(cat)

        questions_right.append(right_string_row_idxquestions)
        questions_wrong.append(wrong_string_row_idxquestions)

        percent_right = (len(right_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 
        percent_wrong = (len(wrong_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 


        percent_questions_right_total.append( str(round(percent_right*100)) +"%")
        percent_questions_wrong_total.append( str(round (percent_wrong*100)) + "%" )

        percent_questions_right_total_raw.append(percent_right)


    individal_question_breakdown = pd.DataFrame(list(zip(all_level_types, questions_right, percent_questions_right_total, percent_questions_right_total_raw, questions_wrong, percent_questions_wrong_total)))
    individal_question_breakdown.columns = ['Level:Type', 'Question #s Correct by All Models', 'Total Correct', 'Total Correct Raw', 'Question #s Incorrect by at least 1 Model',
    'Total Incorrect']
    individal_question_breakdown = individal_question_breakdown.merge(percent_each_leveltype, left_on="Level:Type", right_on="Level:Type")
    individal_question_breakdown = individal_question_breakdown.sort_values(by=['Total Correct Raw'], ascending=False)
    individal_question_breakdown = individal_question_breakdown[individal_question_breakdown["Total Correct Raw"] > 0]
    individal_question_breakdown = individal_question_breakdown.drop(columns=["Total Correct Raw"])
    display(individal_question_breakdown)
    return individal_question_breakdown

def get_competition_math_individual_breakdown_all_incorrect(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]
    
    level_types = all_results['level:type'].unique()

    all_level_types = []
    questions_right = []
    questions_wrong = []

    percent_questions_right_total = []
    percent_questions_wrong_total = []

    percent_questions_right_total_raw = []

    for cat in level_types:
        right_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["all_incorrect"] == True)) ]["row_index"]
        right_string_row_idxquestions = ", ".join(map(str, right_row_indices_of_questions))

        wrong_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["all_incorrect"] == False)) ]["row_index"]
        wrong_string_row_idxquestions = ", ".join(map(str, wrong_row_indices_of_questions))
        
        all_level_types.append(cat)

        questions_right.append(right_string_row_idxquestions)
        questions_wrong.append(wrong_string_row_idxquestions)

        percent_right = (len(right_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 
        percent_wrong = (len(wrong_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 


        percent_questions_right_total.append( str(round(percent_right*100)) +"%")
        percent_questions_wrong_total.append( str(round (percent_wrong*100)) + "%" )

        percent_questions_right_total_raw.append(percent_right)


    individal_question_breakdown = pd.DataFrame(list(zip(all_level_types, questions_right, percent_questions_right_total, percent_questions_right_total_raw, questions_wrong, percent_questions_wrong_total)))
    individal_question_breakdown.columns = ['Level:Type', 'Question #s Incorrect by All Models', 'Total Incorrect', 'Total Incorrect Raw', 'Question #s Correct by at least 1 Model',
    'Total Correct']
    individal_question_breakdown = individal_question_breakdown.merge(percent_each_leveltype, left_on="Level:Type", right_on="Level:Type")
    individal_question_breakdown = individal_question_breakdown.sort_values(by=['Total Incorrect Raw'], ascending=False)
    individal_question_breakdown = individal_question_breakdown[individal_question_breakdown["Total Incorrect Raw"] > 0]
    individal_question_breakdown = individal_question_breakdown.drop(columns=["Total Incorrect Raw"])
    display(individal_question_breakdown)
    return individal_question_breakdown

def get_all_incorrect_vs_1_model_correct(all_results, test_dataset_df):
    df = get_competition_math_individual_breakdown_all_incorrect(all_results, test_dataset_df)

    percent_incorrect = [float(p.strip('%')) for p in df["Total Incorrect"]]
    percent_correct = [float(p.strip('%')) for p in df["Total Correct"]]
    df["Percent Correct"] = percent_correct
    df["Percent Incorrect"] = percent_incorrect
    df = df[["Level:Type", "Percent Incorrect", "Percent Correct"]]

    # Increase figure size and adjust spacing
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust width and height

    # Plot stacked bar chart
    df.plot.bar(x="Level:Type", stacked=True, ax=ax, width=0.7)  # Adjust width for spacing

    # Rotate and align x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add bar labels for percentages
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center', size=8)

    # Adjust title and axis labels for better readability
    ax.set_title("All Incorrect vs At least 1 Model Correct", fontsize=16, pad=20)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)

    # Add grid for clarity
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to fit everything
    plt.tight_layout()
    return plt

def get_competition_math_individual_breakdown_3correct(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]
    
    level_types = all_results['level:type'].unique()

    all_level_types = []
    questions_right = []
    questions_wrong = []

    percent_questions_right_total = []
    percent_questions_wrong_total = []

    percent_questions_right_total_raw = []

    for cat in level_types:
        right_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["at_least_3_correct"] == True)) ]["row_index"]
        right_string_row_idxquestions = ", ".join(map(str, right_row_indices_of_questions))

        wrong_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["at_least_3_correct"] == False)) ]["row_index"]
        wrong_string_row_idxquestions = ", ".join(map(str, wrong_row_indices_of_questions))
        
        all_level_types.append(cat)

        questions_right.append(right_string_row_idxquestions)
        questions_wrong.append(wrong_string_row_idxquestions)

        percent_right = (len(right_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 
        percent_wrong = (len(wrong_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 


        percent_questions_right_total.append( str(round(percent_right*100)) +"%")
        percent_questions_wrong_total.append( str(round (percent_wrong*100)) + "%" )

        percent_questions_right_total_raw.append(percent_right)


    individal_question_breakdown = pd.DataFrame(list(zip(all_level_types, questions_right, percent_questions_right_total, percent_questions_right_total_raw, questions_wrong, percent_questions_wrong_total)))
    individal_question_breakdown.columns = ['Level:Type', 'Question #s Correct by 3 Models', 'Total Correct', 'Total Correct Raw', 'Question #s Incorrect by at least 1 Model',
    'Total Incorrect']
    individal_question_breakdown = individal_question_breakdown.merge(percent_each_leveltype, left_on="Level:Type", right_on="Level:Type")
    individal_question_breakdown = individal_question_breakdown.sort_values(by=['Total Correct Raw'], ascending=False)
    individal_question_breakdown = individal_question_breakdown[individal_question_breakdown["Total Correct Raw"] > 0]
    individal_question_breakdown = individal_question_breakdown.drop(columns=["Total Correct Raw"])
    display(individal_question_breakdown)
    return individal_question_breakdown

def plot_competition_math_individual_breakdown_3correct(all_results, test_dataset_df):
    df = get_competition_math_individual_breakdown_3correct(all_results, test_dataset_df)
    percent_incorrect = [float(p.strip('%')) for p in df["Total Incorrect"]]
    percent_correct = [float(p.strip('%')) for p in df["Total Correct"]]
    df["Percent Correct"] = percent_correct
    df["Percent Incorrect"] = percent_incorrect
    df = df[["Level:Type", "Percent Incorrect", "Percent Correct"]]

    # Increase figure size and adjust spacing
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust width and height

    # Plot stacked bar chart
    df.plot.bar(x="Level:Type", stacked=True, ax=ax, width=0.7)  # Adjust width for spacing

    # Rotate and align x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add bar labels for percentages
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center', size=8)

    # Adjust title and axis labels for better readability
    ax.set_title("All 3 Models Correct", fontsize=16, pad=20)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)

    # Add grid for clarity
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to fit everything
    plt.tight_layout()

    return plt

    # Show plot
    #plt.show()

def get_competition_math_individual_breakdown_2correct(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]
    
    level_types = all_results['level:type'].unique()

    all_level_types = []
    questions_right = []
    questions_wrong = []

    percent_questions_right_total = []
    percent_questions_wrong_total = []

    percent_questions_right_total_raw = []

    for cat in level_types:
        right_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["at_least_2_correct"] == True)) ]["row_index"]
        right_string_row_idxquestions = ", ".join(map(str, right_row_indices_of_questions))

        wrong_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["at_least_2_correct"] == False)) ]["row_index"]
        wrong_string_row_idxquestions = ", ".join(map(str, wrong_row_indices_of_questions))
        
        all_level_types.append(cat)

        questions_right.append(right_string_row_idxquestions)
        questions_wrong.append(wrong_string_row_idxquestions)

        percent_right = (len(right_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 
        percent_wrong = (len(wrong_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 


        percent_questions_right_total.append( str(round(percent_right*100)) +"%")
        percent_questions_wrong_total.append( str(round (percent_wrong*100)) + "%" )

        percent_questions_right_total_raw.append(percent_right)


    individal_question_breakdown = pd.DataFrame(list(zip(all_level_types, questions_right, percent_questions_right_total, percent_questions_right_total_raw, questions_wrong, percent_questions_wrong_total)))
    individal_question_breakdown.columns = ['Level:Type', 'Question #s Correct by 3 Models', 'Total Correct', 'Total Correct Raw', 'Question #s Incorrect by at least 1 Model',
    'Total Incorrect']
    individal_question_breakdown = individal_question_breakdown.merge(percent_each_leveltype, left_on="Level:Type", right_on="Level:Type")
    individal_question_breakdown = individal_question_breakdown.sort_values(by=['Total Correct Raw'], ascending=False)
    individal_question_breakdown = individal_question_breakdown[individal_question_breakdown["Total Correct Raw"] > 0]
    individal_question_breakdown = individal_question_breakdown.drop(columns=["Total Correct Raw"])
    display(individal_question_breakdown)
    return individal_question_breakdown

def plot_competition_math_individual_breakdown_2correct(all_results, test_dataset_df):
    df = get_competition_math_individual_breakdown_2correct(all_results, test_dataset_df)
    percent_incorrect = [float(p.strip('%')) for p in df["Total Incorrect"]]
    percent_correct = [float(p.strip('%')) for p in df["Total Correct"]]
    df["Percent Correct"] = percent_correct
    df["Percent Incorrect"] = percent_incorrect
    df = df[["Level:Type", "Percent Incorrect", "Percent Correct"]]

    # Increase figure size and adjust spacing
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust width and height

    # Plot stacked bar chart
    df.plot.bar(x="Level:Type", stacked=True, ax=ax, width=0.7)  # Adjust width for spacing

    # Rotate and align x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add bar labels for percentages
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center', size=8)

    # Adjust title and axis labels for better readability
    ax.set_title("All 2 Models Correct", fontsize=16, pad=20)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)

    # Add grid for clarity
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to fit everything
    plt.tight_layout()

    return plt

    # Show plot
    #plt.show()   

def get_competition_math_individual_breakdown_1correct(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]
    
    level_types = all_results['level:type'].unique()

    all_level_types = []
    questions_right = []
    questions_wrong = []

    percent_questions_right_total = []
    percent_questions_wrong_total = []

    percent_questions_right_total_raw = []

    for cat in level_types:
        right_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["at_least_1_correct"] == True)) ]["row_index"]
        right_string_row_idxquestions = ", ".join(map(str, right_row_indices_of_questions))

        wrong_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["at_least_1_correct"] == False)) ]["row_index"]
        wrong_string_row_idxquestions = ", ".join(map(str, wrong_row_indices_of_questions))
        
        all_level_types.append(cat)

        questions_right.append(right_string_row_idxquestions)
        questions_wrong.append(wrong_string_row_idxquestions)

        percent_right = (len(right_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 
        percent_wrong = (len(wrong_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 


        percent_questions_right_total.append( str(round(percent_right*100)) +"%")
        percent_questions_wrong_total.append( str(round (percent_wrong*100)) + "%" )

        percent_questions_right_total_raw.append(percent_right)


    individal_question_breakdown = pd.DataFrame(list(zip(all_level_types, questions_right, percent_questions_right_total, percent_questions_right_total_raw, questions_wrong, percent_questions_wrong_total)))
    individal_question_breakdown.columns = ['Level:Type', 'Question #s Correct by 1 Models', 'Total Correct', 'Total Correct Raw', 'Question #s Incorrect',
    'Total Incorrect']
    individal_question_breakdown = individal_question_breakdown.merge(percent_each_leveltype, left_on="Level:Type", right_on="Level:Type")
    individal_question_breakdown = individal_question_breakdown.sort_values(by=['Total Correct Raw'], ascending=False)
    individal_question_breakdown = individal_question_breakdown[individal_question_breakdown["Total Correct Raw"] > 0]
    individal_question_breakdown = individal_question_breakdown.drop(columns=["Total Correct Raw"])
    display(individal_question_breakdown)
    return individal_question_breakdown

def plot_competition_math_individual_breakdown_1correct(all_results, test_dataset_df):
    df = get_competition_math_individual_breakdown_1correct(all_results, test_dataset_df)
    percent_incorrect = [float(p.strip('%')) for p in df["Total Incorrect"]]
    percent_correct = [float(p.strip('%')) for p in df["Total Correct"]]
    df["Percent Correct"] = percent_correct
    df["Percent Incorrect"] = percent_incorrect
    df = df[["Level:Type", "Percent Incorrect", "Percent Correct"]]

    # Increase figure size and adjust spacing
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust width and height

    # Plot stacked bar chart
    df.plot.bar(x="Level:Type", stacked=True, ax=ax, width=0.7)  # Adjust width for spacing

    # Rotate and align x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add bar labels for percentages
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center', size=8)

    # Adjust title and axis labels for better readability
    ax.set_title("1 Model Correct", fontsize=16, pad=20)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)

    # Add grid for clarity
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to fit everything
    plt.tight_layout()

    return plt

    # Show plot
    #plt.show()

def get_competition_math_individual_breakdown_llama(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]
    
    level_types = all_results['level:type'].unique()

    all_level_types = []
    questions_right = []
    questions_wrong = []

    percent_questions_right_total = []
    percent_questions_wrong_total = []

    percent_questions_right_total_raw = []

    for cat in level_types:
        right_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["llama3_is_correct"] == True)) ]["row_index"]
        right_string_row_idxquestions = ", ".join(map(str, right_row_indices_of_questions))

        wrong_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["llama3_is_correct"] == False)) ]["row_index"]
        wrong_string_row_idxquestions = ", ".join(map(str, wrong_row_indices_of_questions))
        
        all_level_types.append(cat)

        questions_right.append(right_string_row_idxquestions)
        questions_wrong.append(wrong_string_row_idxquestions)

        percent_right = (len(right_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 
        percent_wrong = (len(wrong_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 


        percent_questions_right_total.append( str(round(percent_right*100)) +"%")
        percent_questions_wrong_total.append( str(round (percent_wrong*100)) + "%" )

        percent_questions_right_total_raw.append(percent_right)


    individal_question_breakdown = pd.DataFrame(list(zip(all_level_types, questions_right, percent_questions_right_total, percent_questions_right_total_raw, questions_wrong, percent_questions_wrong_total)))
    individal_question_breakdown.columns = ['Level:Type', 'Question #s Correct by Llama Models', 'Total Correct', 'Total Correct Raw', 'Question #s Incorrect',
    'Total Incorrect']
    individal_question_breakdown = individal_question_breakdown.merge(percent_each_leveltype, left_on="Level:Type", right_on="Level:Type")
    individal_question_breakdown = individal_question_breakdown.sort_values(by=['Total Correct Raw'], ascending=False)
    individal_question_breakdown = individal_question_breakdown[individal_question_breakdown["Total Correct Raw"] > 0]
    individal_question_breakdown = individal_question_breakdown.drop(columns=["Total Correct Raw"])
    display(individal_question_breakdown)
    return individal_question_breakdown

def plot_llama_only_correct(all_results, test_dataset_df):
    df = get_competition_math_individual_breakdown_llama(all_results, test_dataset_df)
    percent_incorrect = [float(p.strip('%')) for p in df["Total Incorrect"]]
    percent_correct = [float(p.strip('%')) for p in df["Total Correct"]]
    df["Percent Correct"] = percent_correct
    df["Percent Incorrect"] = percent_incorrect
    df = df[["Level:Type", "Percent Incorrect", "Percent Correct"]]

    # Increase figure size and adjust spacing
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust width and height

    # Plot stacked bar chart
    df.plot.bar(x="Level:Type", stacked=True, ax=ax, width=0.7)  # Adjust width for spacing

    # Rotate and align x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add bar labels for percentages
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center', size=8)

    # Adjust title and axis labels for better readability
    ax.set_title("Lllama Only Correct", fontsize=16, pad=20)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)

    # Add grid for clarity
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to fit everything
    plt.tight_layout()

    return plt

    # Show plot
    #plt.show()

def get_competition_math_individual_breakdown_llema(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]
    
    level_types = all_results['level:type'].unique()

    all_level_types = []
    questions_right = []
    questions_wrong = []

    percent_questions_right_total = []
    percent_questions_wrong_total = []

    percent_questions_right_total_raw = []

    for cat in level_types:
        right_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["llema_is_correct"] == True)) ]["row_index"]
        right_string_row_idxquestions = ", ".join(map(str, right_row_indices_of_questions))

        wrong_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["llema_is_correct"] == False)) ]["row_index"]
        wrong_string_row_idxquestions = ", ".join(map(str, wrong_row_indices_of_questions))
        
        all_level_types.append(cat)

        questions_right.append(right_string_row_idxquestions)
        questions_wrong.append(wrong_string_row_idxquestions)

        percent_right = (len(right_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 
        percent_wrong = (len(wrong_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 


        percent_questions_right_total.append( str(round(percent_right*100)) +"%")
        percent_questions_wrong_total.append( str(round (percent_wrong*100)) + "%" )

        percent_questions_right_total_raw.append(percent_right)


    individal_question_breakdown = pd.DataFrame(list(zip(all_level_types, questions_right, percent_questions_right_total, percent_questions_right_total_raw, questions_wrong, percent_questions_wrong_total)))
    individal_question_breakdown.columns = ['Level:Type', 'Question #s Correct by Llama Models', 'Total Correct', 'Total Correct Raw', 'Question #s Incorrect',
    'Total Incorrect']
    individal_question_breakdown = individal_question_breakdown.merge(percent_each_leveltype, left_on="Level:Type", right_on="Level:Type")
    individal_question_breakdown = individal_question_breakdown.sort_values(by=['Total Correct Raw'], ascending=False)
    individal_question_breakdown = individal_question_breakdown[individal_question_breakdown["Total Correct Raw"] > 0]
    individal_question_breakdown = individal_question_breakdown.drop(columns=["Total Correct Raw"])
    display(individal_question_breakdown)
    return individal_question_breakdown

def plot_competition_math_individual_breakdown_llema(all_results, test_dataset_df):
    df =  get_competition_math_individual_breakdown_llema(all_results, test_dataset_df)
    percent_incorrect = [float(p.strip('%')) for p in df["Total Incorrect"]]
    percent_correct = [float(p.strip('%')) for p in df["Total Correct"]]
    df["Percent Correct"] = percent_correct
    df["Percent Incorrect"] = percent_incorrect
    df = df[["Level:Type", "Percent Incorrect", "Percent Correct"]]

    # Increase figure size and adjust spacing
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust width and height

    # Plot stacked bar chart
    df.plot.bar(x="Level:Type", stacked=True, ax=ax, width=0.7)  # Adjust width for spacing

    # Rotate and align x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add bar labels for percentages
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center', size=8)

    # Adjust title and axis labels for better readability
    ax.set_title("Llema Only Correct", fontsize=16, pad=20)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)

    # Add grid for clarity
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to fit everything
    plt.tight_layout()

    return plt

    # Show plot
    #plt.show()

def get_competition_math_individual_breakdown_qwen(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]
    
    level_types = all_results['level:type'].unique()

    all_level_types = []
    questions_right = []
    questions_wrong = []

    percent_questions_right_total = []
    percent_questions_wrong_total = []

    percent_questions_right_total_raw = []

    for cat in level_types:
        right_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["qwen_is_correct"] == True)) ]["row_index"]
        right_string_row_idxquestions = ", ".join(map(str, right_row_indices_of_questions))

        wrong_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["qwen_is_correct"] == False)) ]["row_index"]
        wrong_string_row_idxquestions = ", ".join(map(str, wrong_row_indices_of_questions))
        
        all_level_types.append(cat)

        questions_right.append(right_string_row_idxquestions)
        questions_wrong.append(wrong_string_row_idxquestions)

        percent_right = (len(right_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 
        percent_wrong = (len(wrong_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 


        percent_questions_right_total.append( str(round(percent_right*100)) +"%")
        percent_questions_wrong_total.append( str(round (percent_wrong*100)) + "%" )

        percent_questions_right_total_raw.append(percent_right)


    individal_question_breakdown = pd.DataFrame(list(zip(all_level_types, questions_right, percent_questions_right_total, percent_questions_right_total_raw, questions_wrong, percent_questions_wrong_total)))
    individal_question_breakdown.columns = ['Level:Type', 'Question #s Correct by Llama Models', 'Total Correct', 'Total Correct Raw', 'Question #s Incorrect',
    'Total Incorrect']
    individal_question_breakdown = individal_question_breakdown.merge(percent_each_leveltype, left_on="Level:Type", right_on="Level:Type")
    individal_question_breakdown = individal_question_breakdown.sort_values(by=['Total Correct Raw'], ascending=False)
    individal_question_breakdown = individal_question_breakdown[individal_question_breakdown["Total Correct Raw"] > 0]
    individal_question_breakdown = individal_question_breakdown.drop(columns=["Total Correct Raw"])
    display(individal_question_breakdown)
    return individal_question_breakdown

def plot_competition_math_individual_breakdown_qwen(all_results, test_dataset_df):
    df = get_competition_math_individual_breakdown_qwen(all_results, test_dataset_df)
    percent_incorrect = [float(p.strip('%')) for p in df["Total Incorrect"]]
    percent_correct = [float(p.strip('%')) for p in df["Total Correct"]]
    df["Percent Correct"] = percent_correct
    df["Percent Incorrect"] = percent_incorrect
    df = df[["Level:Type", "Percent Incorrect", "Percent Correct"]]

    # Increase figure size and adjust spacing
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust width and height

    # Plot stacked bar chart
    df.plot.bar(x="Level:Type", stacked=True, ax=ax, width=0.7)  # Adjust width for spacing

    # Rotate and align x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add bar labels for percentages
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center', size=8)

    # Adjust title and axis labels for better readability
    ax.set_title("Qwen Only Correct", fontsize=16, pad=20)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)

    # Add grid for clarity
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to fit everything
    plt.tight_layout()

    return plt

    # Show plot
    #plt.show()

def get_competition_math_individual_breakdown_mistral(all_results, test_dataset_df):
    test_dataset_df["Level:Type"] = test_dataset_df["level"] + ": " + test_dataset_df["type"]
    percent_each_leveltype = pd.DataFrame(test_dataset_df["Level:Type"].value_counts())
    percent_each_leveltype['count'] = round((percent_each_leveltype['count'] / percent_each_leveltype['count'].sum()) * 100, 1)
    percent_each_leveltype.columns = ["Percent of Level:Type in Dataset"]
    
    level_types = all_results['level:type'].unique()

    all_level_types = []
    questions_right = []
    questions_wrong = []

    percent_questions_right_total = []
    percent_questions_wrong_total = []

    percent_questions_right_total_raw = []

    for cat in level_types:
        right_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["mistral_is_correct"] == True)) ]["row_index"]
        right_string_row_idxquestions = ", ".join(map(str, right_row_indices_of_questions))

        wrong_row_indices_of_questions = all_results[ ((all_results['level:type'] == cat) & (all_results["mistral_is_correct"] == False)) ]["row_index"]
        wrong_string_row_idxquestions = ", ".join(map(str, wrong_row_indices_of_questions))
        
        all_level_types.append(cat)

        questions_right.append(right_string_row_idxquestions)
        questions_wrong.append(wrong_string_row_idxquestions)

        percent_right = (len(right_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 
        percent_wrong = (len(wrong_row_indices_of_questions)) / (len(right_row_indices_of_questions) + len(wrong_row_indices_of_questions)) 


        percent_questions_right_total.append( str(round(percent_right*100)) +"%")
        percent_questions_wrong_total.append( str(round (percent_wrong*100)) + "%" )

        percent_questions_right_total_raw.append(percent_right)


    individal_question_breakdown = pd.DataFrame(list(zip(all_level_types, questions_right, percent_questions_right_total, percent_questions_right_total_raw, questions_wrong, percent_questions_wrong_total)))
    individal_question_breakdown.columns = ['Level:Type', 'Question #s Correct by Llama Models', 'Total Correct', 'Total Correct Raw', 'Question #s Incorrect',
    'Total Incorrect']
    individal_question_breakdown = individal_question_breakdown.merge(percent_each_leveltype, left_on="Level:Type", right_on="Level:Type")
    individal_question_breakdown = individal_question_breakdown.sort_values(by=['Total Correct Raw'], ascending=False)
    individal_question_breakdown = individal_question_breakdown[individal_question_breakdown["Total Correct Raw"] > 0]
    individal_question_breakdown = individal_question_breakdown.drop(columns=["Total Correct Raw"])
    display(individal_question_breakdown)
    return individal_question_breakdown

def plot_competition_math_individual_breakdown_mistral(all_results, test_dataset_df):
    df = get_competition_math_individual_breakdown_mistral(all_results, test_dataset_df)
    percent_incorrect = [float(p.strip('%')) for p in df["Total Incorrect"]]
    percent_correct = [float(p.strip('%')) for p in df["Total Correct"]]
    df["Percent Correct"] = percent_correct
    df["Percent Incorrect"] = percent_incorrect
    df = df[["Level:Type", "Percent Incorrect", "Percent Correct"]]

    # Increase figure size and adjust spacing
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust width and height

    # Plot stacked bar chart
    df.plot.bar(x="Level:Type", stacked=True, ax=ax, width=0.7)  # Adjust width for spacing

    # Rotate and align x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add bar labels for percentages
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f%%', label_type='center', size=8)

    # Adjust title and axis labels for better readability
    ax.set_title("Mistral Only Correct", fontsize=16, pad=20)
    ax.set_ylabel("Percentage", fontsize=14)
    ax.set_xlabel("Category", fontsize=14)

    # Add grid for clarity
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to fit everything
    plt.tight_layout()

    return plt

    # Show plot
    #plt.show()

def print_competition_math_question_llama_output(row_index, llama_file_path,  all_results, test_dataset_df):
    display(all_results[all_results["row_index"] == row_index][["llama_solution", "llema_solution", "qwen_solution", "mistral_solution"]])

    display(all_results[all_results["row_index"] == row_index])

    df = pd.read_csv(llama_file_path)
    print(list(df[df["row_index"] == row_index]['Output'])[0])

    display(test_dataset_df[test_dataset_df["row_index"] == row_index])

    print(list(test_dataset_df[test_dataset_df["row_index"] == row_index ]["problem"])[0])

    print(list(test_dataset_df[test_dataset_df["row_index"] == row_index ]["solution"])[0])

def print_competition_math_list_llama_questions(all_results, is_correct):
    with np.printoptions(threshold=np.inf):
        print(np.array(all_results[all_results["llama3_is_correct"] == is_correct]["row_index"]))