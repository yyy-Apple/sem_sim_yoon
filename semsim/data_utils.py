import json
import jsonlines
import os


def write_summary_json():
    """
    for each unique text id, write the json file for its summaries in a json file
    {
        'id': '...',
        'referenceSummary': '...',
        'summaryList': '[...]',
        'scoreList': '[...]'
    }
    :return: write json files under the summary folder, name like "id_summary.json"
    """
    with open("./data/cdm/sorted_scores.json", "r", encoding="utf8") as f1:
        summaries = json.loads(f1.read())

        # for each text id
        for element in summaries:
            textSummaryDict = {}
            summaryDictList = summaries.get(element)
            referenceSummary = summaryDictList[0].get('ref')
            id = summaryDictList[0].get('id')
            summaryList = []
            scoreList = []  # overall score
            for summaryDict in summaryDictList:
                summaryList.append(summaryDict.get('sys_summ'))
                scoreList.append(summaryDict.get('scores').get('overall'))

            # update the textSummaryDict
            textSummaryDict.update({"id": id, "referenceSummary": referenceSummary,
                                    "summaryList": summaryList, "scoreList": scoreList})
            filepath = "./summary/" + id + "_summary.json"
            with open(filepath, "w", encoding="utf8") as f2:
                f2.write(json.dumps(textSummaryDict))


def write_text_json():
    """
    for each unique text id, write the json file for its text in a json file
    {
        'id': '...',
        'text': '...'
    }
    :return: write json files under the text folder, name like 'id_text.json'
    """
    with jsonlines.open("./data/cdm/raw_json/articles.jsonl", "r") as reader:
        for obj in reader:
            textDict = {}
            id = obj.get('id')
            text = obj.get('text')
            textDict.update({"id": id, "text": text})
            filepath = "./text/" + id + "_text.json"
            with open(filepath, "w", encoding="utf8") as f:
                f.write(json.dumps(textDict))


def write_text_and_summary():
    """
    Since not all the text has corresponding summaries, use the summary folder
    to construct id_text_and_summary.json file
    {
        'id': '...'
        'text': '...'
        'referenceSummary': '...'
        'summaryList': '[...]'
        'scoreList': '[...]'
    }
    :return: write json files under the text_and_summary folder, name like 'id_text_and_summary.json'
    """
    summaryList = os.listdir("summary")
    for summary in summaryList:
        summaryFile = "./summary/" + summary
        textSummaryDict = {}
        print(summaryFile)
        with open(summaryFile, "r", encoding="utf8") as f:
            summaryContent = json.loads(f.read())
            textSummaryDict.update({"id": summaryContent.get('id'),
                                    "referenceSummary": summaryContent.get('referenceSummary'),
                                    "summaryList": summaryContent.get('summaryList'),
                                    "scoreList": summaryContent.get("scoreList")})
        # retrieve text information from the text folder
        textFile = "./text/" + summary[: -13] + '_text.json'
        with open(textFile, "r", encoding="utf8") as f:
            textContent = json.loads(f.read())
            textSummaryDict.update({"text": textContent.get('text')})

        textSummaryFile = "./text_and_summary/" + summary[: -13] + '_text_and_summary.json'
        with open(textSummaryFile, "w", encoding="utf8") as f:
            f.write(json.dumps(textSummaryDict))


if __name__ == '__main__':
    write_text_json()
    write_summary_json()
    write_text_and_summary()
