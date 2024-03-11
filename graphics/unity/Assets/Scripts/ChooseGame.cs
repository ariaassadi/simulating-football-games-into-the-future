using System.Collections;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;

public class ChooseGame : MonoBehaviour
{
    private Schedule[] schedule;
    [SerializeField] private GameObject gameSelectionOptionPrefab;

    [SerializeField] private GameObject content;

    public void GetSchedule()
    {
        schedule = DatabaseManager.query_schedule_db($"SELECT match_id, home_team_name, away_team_name FROM schedule");

        if (schedule == null)
        {
            Debug.LogError("No schedule found");
            return;
        }
        float gameSelectionOptionPrefabHeight = gameSelectionOptionPrefab.GetComponent<RectTransform>().sizeDelta.y;
        Vector3 contentPosition = content.GetComponent<RectTransform>().position;
        contentPosition.y = schedule.Length * (gameSelectionOptionPrefabHeight + 10);
        for (int i = 0; i < schedule.Length; i++)
        {
            // public static Object Instantiate(Object original, Vector3 position, Quaternion rotation, Transform parent); 
            Vector3 position = new Vector3(contentPosition.x, contentPosition.y + i * (gameSelectionOptionPrefabHeight + 10), contentPosition.z);
            GameObject gameSelectionOption = Instantiate(gameSelectionOptionPrefab, position, Quaternion.identity, content.transform);
            gameSelectionOption.GetComponentInChildren<TMPro.TextMeshProUGUI>().text = $"{schedule[i].HomeTeamName} vs {schedule[i].AwayTeamName}";
        }

    }

}
