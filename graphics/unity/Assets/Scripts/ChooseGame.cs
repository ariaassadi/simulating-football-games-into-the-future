using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;

public class ChooseGame : MonoBehaviour
{
    private Schedule[] schedule;
    [SerializeField] private GameObject gameOptionPrefab;

    [SerializeField] private GameObject HorizontalPanelPrefab;

    [SerializeField] private GameObject content;

    private GameObject uiManager;

    private GameObject gameManager;

    private void Start()
    {
        gameManager = GameObject.Find("GameManager");
        uiManager = GameObject.Find("UIManager");
        if (gameManager == null)
        {
            Debug.LogError("GameManager is not assigned.");
            return;
        }
        if (uiManager == null)
        {
            Debug.LogError("UIManager is not assigned.");
            return;
        }
    }

    public void GetSchedule()
    {
        schedule = DatabaseManager.query_schedule_db($"SELECT * FROM schedule");

        if (schedule == null)
        {
            Debug.LogError("No schedule found");
            return;
        }

        GameObject horizontalPanel;
        GameObject g;

        Debug.Log(schedule.Length);
        if (schedule.Length % 2 == 0)
        {
            for (int i = 0; i < schedule.Length; i += 2)
            {
                // public static Object Instantiate(Object original, Vector3 position, Quaternion rotation, Transform parent); 

                horizontalPanel = Instantiate(HorizontalPanelPrefab, content.transform, false);

                for (int j = i; j < i + 2; j++)
                {
                    g = Instantiate(gameOptionPrefab, horizontalPanel.transform, false);
                    AddGameInfo(schedule[j], g);
                }
            }
        }
        else
        {
            int lastGame = schedule.Length - 1;
            for (int i = 0; i < lastGame; i++)
            {
                horizontalPanel = Instantiate(HorizontalPanelPrefab, content.transform, false);

                for (int j = i; j < i + 2; j++)
                {
                    g = Instantiate(gameOptionPrefab, horizontalPanel.transform, false);
                    AddGameInfo(schedule[j], g);
                }
            }
            horizontalPanel = Instantiate(HorizontalPanelPrefab, content.transform, false);
            g = Instantiate(gameOptionPrefab, horizontalPanel.transform, false);
            AddGameInfo(schedule[lastGame], g);
        }
    }

    private void AddGameInfo(Schedule game, GameObject g)
    {
        g.GetComponentInChildren<TMPro.TextMeshProUGUI>().text = $"{game.HomeTeamName} vs {game.AwayTeamName}";
        g.name = $"{game.HomeTeamName} vs {game.AwayTeamName}";
        // g.GetComponent<Button>().onClick.AddListener(() => uiManager.GetComponent<MenuManager>().ShowLoadingScreen());
        g.GetComponent<Button>().onClick.AddListener(() => LoadGame(game));
    }

    private async void LoadGame(Schedule game)
    {
        // Show loading screen
        uiManager.GetComponent<MenuManager>().ShowLoadingScreen();

        // Load the game asynchronously
        bool success = await LoadGameAsync(game);

        // Check if loading was successful
        if (success)
        {
            // Show the game
            uiManager.GetComponent<MenuManager>().ShowGame();
        }
        else
        {
            // Show an error message or handle failure
            Debug.LogError("Failed to load game");
            // Optionally, you can hide the loading screen here as well
            // uiManager.GetComponent<MenuManager>().HideLoadingScreen();
        }
    }

    private async Task<bool> LoadGameAsync(Schedule game)
    {
        // Perform the loading asynchronously
        bool success = await gameManager.GetComponent<GameManager>().LoadGameAsync(game);

        // Return the result
        return success;
    }

}
