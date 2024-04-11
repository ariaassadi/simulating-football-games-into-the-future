using UnityEngine;
using UnityEngine.UI;
using System.Data;
using System;
using System.Threading.Tasks;
using TMPro;
using Eggs;

public class GameManager : MonoBehaviour
{

    // Prefabs for the players and the ball
    [SerializeField] private GameObject playerHomePrefab;
    [SerializeField] private GameObject playerAwayPrefab;
    [SerializeField] private GameObject playerPrefab;
    [SerializeField] private GameObject ballPrefab;

    // The game objects for the home team, away team and the ball
    [SerializeField] private GameObject homeTeam;
    [SerializeField] private GameObject awayTeam;
    [SerializeField] private GameObject ball;

    // The play and pause icons
    [SerializeField] private Texture2D playIcon;
    [SerializeField] private Texture2D pauseIcon;

    // The play and pause button
    [SerializeField] private GameObject playPauseButton;

    [SerializeField] private GameObject timeSlider;

    [SerializeField] private GameObject homeTeamNameShort;
    [SerializeField] private GameObject awayTeamNameShort;

    // Scripts
    [SerializeField] private GameObject gameUI;
    private GameObject uiManager;
    private ToolManager toolManager;

    private int startFrame;
    private int endFrame;
    private int period;

    private int currentFrameStartIndex;
    private int currentFrameEndIndex;
    private int currentFrameNr;

    private Game[] playerData;
    private Game[] frameData;

    private Schedule gameInfo;

    private PlayerData[] playerPositions;
    // Line renderers for the offside lines
    [SerializeField] private LineRenderer offsideLineHome;
    [SerializeField] private LineRenderer offsideLineAway;
    private float homeOffsideLine = 0;
    private float awayOffsideLine = 0;
    [SerializeField] private float offsideBrightness = 0.5f;

    float timer = 0f;
    float interval = 0.04f; // 40 milliseconds in seconds, the interval between frames
    private bool isPlaying = false;
    private bool changingGame = false; // True when changing the game, maybe unnecessary

    private string homeTeamColor;
    private string awayTeamColor;

    private int[][] framesStartAndEndIndex;


    private void Start()
    {
        // Find the UIManager object in the scene
        uiManager = GameObject.Find("UIManager");
        if (uiManager == null)
        {
            Debug.LogError("UIManager is not assigned.");
            return;
        }
        toolManager = GameObject.Find("ToolManager").GetComponent<ToolManager>();
        if (toolManager == null)
        {
            Debug.LogError("ToolManager is not assigned.");
            return;
        }

        offsideLineHome.startWidth = 0.2f;
        offsideLineHome.endWidth = 0.2f;
        offsideLineHome.positionCount = 2;
        offsideLineHome.material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        offsideLineHome.material.color = Color.red;

        offsideLineAway.startWidth = 0.2f;
        offsideLineAway.endWidth = 0.2f;
        offsideLineAway.positionCount = 2;
        offsideLineAway.material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
        offsideLineAway.material.color = Color.red;

    }


    void FixedUpdate()
    {
        // If playing and not currently changing the game, execute Play method
        if (isPlaying && !changingGame)
        {
            Play();
        }
    }

    // Asynchronously load game data from the database using match ID
    public async Task<bool> LoadGameAsync(string match_id, int period)
    {
        this.period = period;
        string pathToDB;
        // Path to the database
        // if platform is Android, use persistent data path, otherwise use streaming assets path
        if (Application.platform == RuntimePlatform.Android && !Application.isEditor)
        {
            pathToDB = Application.persistentDataPath + "/2sec_demo.sqlite";
        }
        else
        {
            pathToDB = Application.streamingAssetsPath + "/2sec_demo.sqlite";
        }
        // SQL query to retrieve player data for a specific match and period

        // string query_tracking = $"SELECT player, x, y, frame, team, orientation, jersey_number, offside, v_x, v_y FROM games WHERE period={period} AND match_id='{match_id}'";
        string query_tracking = $"SELECT player, x, y, frame, team, orientation, jersey_number, offside, v_x, v_y FROM games WHERE frame>=0 AND frame<=5000 AND period={period} AND match_id='{match_id}'";

        // string query_frames = $"SELECT frame, objects_tracked FROM games WHERE period={period} AND match_id='{match_id}' GROUP BY frame";
        string query_frames = $"SELECT frame, objects_tracked FROM games WHERE frame>=0 AND frame<=5000 AND period={period} AND match_id='{match_id}' GROUP BY frame";


        Debug.Log($"Select: {query_tracking} END");
        Debug.Log($"Select: {query_frames} END");

        // Retrieve players data asynchronously
        Task<Game[]> playerDataTask = Task.Run(() => DatabaseManager.query_db(pathToDB, query_tracking));
        playerData = await playerDataTask;

        // Retrieve frames data asynchronously
        Task<Game[]> frameDataTask = Task.Run(() => DatabaseManager.query_db(pathToDB, query_frames));
        frameData = await frameDataTask;


        Debug.Log("Number of rows: " + playerData.Length);
        Debug.Log("Number of frames: " + frameData.Length);

        if (playerData == null || playerData.Length == 0)
        {
            Debug.Log("No players found");
            // error
            return false; // Loading failed
        }
        if (frameData == null || frameData.Length == 0)
        {
            Debug.Log("No frames found");
            // error
            return false; // Loading failed
        }
        CalculateFramesStartAndEndIndex();
        startFrame = frameData[0].Frame;
        endFrame = frameData[frameData.Length - 1].Frame;
        Debug.Log("Start frame: " + startFrame);
        Debug.Log("End frame: " + endFrame);
        // Update timeslider
        timeSlider.GetComponent<TimeSlider>().UpdateTimeSlider(startFrame, endFrame, period);

        if (playerData != null && frameData != null && frameData.Length > 0)
        {
            // Loop through frames to spawn objects
            playerPositions = new PlayerData[frameData[0].ObjectsTracked];
            for (int i = 0; i < frameData[0].ObjectsTracked; i++)
            {
                SpawnObject(playerData[i]);
                playerPositions[i] = new PlayerData();
                playerPositions[i].x = playerData[i].X;
                playerPositions[i].y = playerData[i].Y;
                playerPositions[i].team = playerData[i].Team;
                playerPositions[i].jersey_number = playerData[i].JerseyNumber;
                // Convert x velocity and y velocity to the combined velocity
                playerPositions[i].v = Mathf.Sqrt(Mathf.Pow(playerData[i].V_X, 2) + Mathf.Pow(playerData[i].V_Y, 2));
                playerPositions[i].orientation = Mathf.Deg2Rad * (playerData[i].Orientation + 90);
            }

            homeTeamColor = gameInfo.HomeTeamColor;
            awayTeamColor = gameInfo.AwayTeamColor;

            // Update the pitch control texture
            toolManager.UpdateSynchronized();


            currentFrameNr = startFrame;
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);

            return true; // Loading successful
        }
        else
        {
            Debug.Log("No frames found");
            return false; // Loading failed
        }
    }

    // Overloaded method to load game using Schedule object
    public async Task<bool> LoadGameAsync(Schedule schedule, int period)
    {
        // Inistialize loading screen
        Debug.Log("Loading game: " + schedule.MatchId + " period: " + period);
        gameInfo = schedule;
        // Remove all objects from the scene
        changingGame = true;
        RemoveObjects();
        changingGame = false;
        // gameUI.GetComponent<TimeOverlay>().Timer(0);
        homeTeamNameShort.GetComponentInChildren<TMP_Text>().text = schedule.HomeTeamNameShort;
        homeTeamNameShort.GetComponentInChildren<Image>().color = Utils.HexToColor(schedule.HomeTeamColor);
        awayTeamNameShort.GetComponentInChildren<TMP_Text>().text = schedule.AwayTeamNameShort;
        awayTeamNameShort.GetComponentInChildren<Image>().color = Utils.HexToColor(schedule.AwayTeamColor);

        return await LoadGameAsync(schedule.MatchId, period);
    }

    private void RemoveObjects()
    {
        // Destroy all child objects of home team, away team, and ball
        DestroyChildren(homeTeam);
        DestroyChildren(awayTeam);
        DestroyChildren(ball);
        LineRenderer lineRenderer = GetComponent<LineRenderer>();
        if (lineRenderer != null)
        {
            Destroy(lineRenderer);
        }
        gameObject.GetComponent<GameTools>().DeselectAllTools();
        Debug.Log("Objects removed");

        // Reset the time slider
        timeSlider.GetComponent<TimeSlider>().ChangeTime(startFrame);

        playerData = null;
        frameData = null;
    }

    private void DestroyChildren(GameObject parent)
    {
        // Destroy all child objects of a parent object
        foreach (Transform child in parent.transform)
        {
            Destroy(child.gameObject);
        }
    }


    private void CalculateFramesStartAndEndIndex()
    {
        // Calculate the start and end index of each frame and store in an array
        int frameStartIndex = 0;
        int frameEndIndex = 0;

        int framesLength = frameData.Length;
        framesStartAndEndIndex = new int[framesLength][];

        for (int i = 0; i < framesLength; i++)
        {
            frameStartIndex = frameEndIndex;
            frameEndIndex += frameData[i].ObjectsTracked;
            framesStartAndEndIndex[i] = new int[] { frameStartIndex, frameEndIndex };
        }
    }

    // Get the start index of a frame
    public int GetFrameStartIndex(int frameNr)
    {
        return framesStartAndEndIndex[NormalizeFrameNr(frameNr)][0];
    }

    // Get the end index of a frame
    public int GetFrameEndIndex(int frameNr)
    {
        return framesStartAndEndIndex[NormalizeFrameNr(frameNr)][1];
    }

    private void Play()
    {
        // Play the game by moving objects according to frame data
        if (NormalizeFrameNr(currentFrameNr) < frameData.Length)
        {
            timer += Time.deltaTime;

            if (timer >= interval)
            {
                timer = 0f;

                MoveObjects();

                if (NormalizeFrameNr(currentFrameNr) + 1 < frameData.Length)
                {
                    currentFrameNr++;
                    currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
                    currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
                }
            }
        }
        else
        {
            Debug.Log("Frame has completed");
        }
    }

    // Toggle play/pause state
    public void PlayPause()
    {
        Debug.Log("PlayPause");
        isPlaying = !isPlaying;
        if (isPlaying)
        {
            playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = pauseIcon;
        }
        else
        {
            playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = playIcon;
        }
    }

    // Set play state to false
    private void SetPlayFalse()
    {
        isPlaying = false;
        playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = playIcon;
    }

    // Fast forward the game
    public void FastForward()
    {
        SetPlayFalse();
        Debug.Log("FastForward");
        if (NormalizeFrameNr(currentFrameNr) + 25 < frameData.Length)
            currentFrameNr += 25;
        else
            currentFrameNr = frameData.Length - 1;

        currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
        currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        MoveObjects();
    }

    // Fast backward the game
    public void FastBackward()
    {
        SetPlayFalse();
        Debug.Log("FastBackward");
        if ((NormalizeFrameNr(currentFrameNr) - 25) > 0)
        {
            Debug.Log("Current frame first if: " + currentFrameNr);
            currentFrameNr -= 25;
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        else
        {
            Debug.Log("Current frame second if: " + currentFrameNr);
            currentFrameNr = 0;
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }

        MoveObjects();
    }

    // Step forward the game by one frame
    public void StepForward()
    {
        SetPlayFalse();
        Debug.Log("StepForward");
        if (NormalizeFrameNr(currentFrameNr) + 1 < frameData.Length)
        {
            currentFrameNr++;
            currentFrameEndIndex += frameData[NormalizeFrameNr(currentFrameNr)].ObjectsTracked;
        }
        else
        {
            currentFrameNr = frameData.Length - 1;
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        MoveObjects();
    }

    // Step backward the game by one frame
    public void StepBackward()
    {
        SetPlayFalse();
        Debug.Log("StepBackward");
        currentFrameNr--;
        if (NormalizeFrameNr(currentFrameNr) < frameData.Length && NormalizeFrameNr(currentFrameNr) > 0)
        {
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        else
        {
            currentFrameNr = startFrame;
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        MoveObjects();
    }

    // Move to a specific frame in the game
    public void MoveTo(int frameNr)
    {
        SetPlayFalse();
        Debug.Log("MoveTo");
        currentFrameNr = frameNr;
        Debug.Log("Current frame (MoveTo): " + currentFrameNr);
        if (NormalizeFrameNr(currentFrameNr) < frameData.Length)
        {
            Debug.Log("Frame found");
            currentFrameStartIndex = GetFrameStartIndex(currentFrameNr);
            currentFrameEndIndex = GetFrameEndIndex(currentFrameNr);
        }
        else
        {
            Debug.Log("Frame not found");
            currentFrameStartIndex = GetFrameStartIndex(endFrame);
            currentFrameEndIndex = GetFrameEndIndex(endFrame);
        }
        MoveObjects();
    }

    // Move objects in the scene according to frame data
    private void MoveObjects()
    {
        Vector3 position;
        Transform playerTransform;
        playerPositions = new PlayerData[currentFrameEndIndex - currentFrameStartIndex];

        Debug.Log("Current frame :" + currentFrameNr);
        for (int currentFrameIndex = currentFrameStartIndex; currentFrameIndex < currentFrameEndIndex; currentFrameIndex++)
        {
            string playerName = playerData[currentFrameIndex].Player;
            int i = currentFrameIndex - currentFrameStartIndex;
            playerPositions[i] = new PlayerData();
            playerPositions[i].x = playerData[currentFrameIndex].X;
            playerPositions[i].y = playerData[currentFrameIndex].Y;
            playerPositions[i].team = playerData[currentFrameIndex].Team;
            playerPositions[i].jersey_number = playerData[currentFrameIndex].JerseyNumber;
            // Convert x velocity and y velocity to the combined velocity
            playerPositions[i].v = Mathf.Sqrt(Mathf.Pow(playerData[currentFrameIndex].V_X, 2) + Mathf.Pow(playerData[currentFrameIndex].V_Y, 2));
            playerPositions[i].orientation = Mathf.Deg2Rad * (playerData[currentFrameIndex].Orientation + 90);
            playerTransform = GameObject.Find(playerName)?.transform;

            if (playerTransform == null)
            {
                Debug.Log("Player not found: " + playerName);
                SpawnObject(playerData[currentFrameIndex]);
            }
            else
            {
                position = playerTransform.position;

                // Move the player to the new position
                position.x = playerData[currentFrameIndex].X;
                position.z = playerData[currentFrameIndex].Y;
                playerTransform.rotation = Quaternion.Euler(0, playerData[currentFrameIndex].Orientation + 90, 0);
                playerTransform.position = position;

                // Show or hide the offside line
                if (playerData[currentFrameIndex].Offside != 0)
                {
                    if (playerData[currentFrameIndex].Team == "home_team")
                    {
                        homeOffsideLine = playerData[currentFrameIndex].Offside;
                        playerTransform.GetComponent<Renderer>().material.color = Utils.HexToColor(gameInfo.HomeTeamColor, offsideBrightness);
                    }
                    else
                    {
                        awayOffsideLine = playerData[currentFrameIndex].Offside;
                        playerTransform.GetComponent<Renderer>().material.color = Utils.HexToColor(gameInfo.AwayTeamColor, offsideBrightness);

                    }
                }
                else if (playerData[currentFrameIndex].Team != "ball")
                {
                    playerTransform.GetComponent<Renderer>().material.color = Utils.HexToColor(playerData[currentFrameIndex].Team == "home_team" ? gameInfo.HomeTeamColor : gameInfo.AwayTeamColor);
                }
            }
        }

        if (homeOffsideLine != 0)
        {
            PrintOffsideLine(homeOffsideLine, "home_team");
        }
        else
        {
            RemoveOffsideLine("home_team");
        }
        if (awayOffsideLine != 0)
        {
            PrintOffsideLine(awayOffsideLine, "away_team");
        }
        else
        {
            RemoveOffsideLine("away_team");
        }

        // Reset the offside line values
        homeOffsideLine = 0;
        awayOffsideLine = 0;

        // Update the pitch control
        toolManager.UpdateSynchronized();

        // Update the time slider
        timeSlider.GetComponent<TimeSlider>().ChangeTime(currentFrameNr);
        // gameUI.GetComponent<TimeOverlay>().Timer(currentFrameNr);
    }

    // Spawn player or ball object
    private void SpawnObject(Game player)
    {
        Vector3 position;
        if (player.Team == "home_team")
        {
            position = new Vector3(player.X, 0, player.Y);
            if (player.Offside != 0)
            {
                homeOffsideLine = player.Offside;
                SpawnPlayer(position, player, homeTeam, gameInfo.HomeTeamColor, offsideBrightness);
            }
            else
                SpawnPlayer(position, player, homeTeam, gameInfo.HomeTeamColor);
        }
        else if (player.Team == "away_team")
        {
            position = new Vector3(player.X, 0, player.Y);
            if (player.Offside != 0)
            {
                awayOffsideLine = player.Offside;
                SpawnPlayer(position, player, awayTeam, gameInfo.AwayTeamColor, offsideBrightness);
            }
            else
                SpawnPlayer(position, player, awayTeam, gameInfo.AwayTeamColor);
        }
        else
        {
            position = new Vector3(player.X, 0.1f, player.Y);
            SpawnBall(position, player);
        }
    }

    // Spawn player object at specified position
    private void SpawnPlayer(Vector3 position, Game player, GameObject team, string teamColor, float brightness = 1.0f)
    {
        GameObject playerObject = Instantiate(playerPrefab, position, Quaternion.Euler(0, player.Orientation, 0), team.transform) as GameObject;
        playerObject.name = player.Player;
        playerObject.tag = "Player";
        playerObject.GetComponent<Renderer>().material.color = Utils.HexToColor(teamColor, brightness);
        playerObject.transform.GetChild(0).gameObject.SetActive(false);
        playerObject.transform.GetChild(1).gameObject.GetComponent<EggUI>().SetJerseyNumber(player.JerseyNumber.ToString());
        playerObject.transform.GetChild(1).gameObject.GetComponent<EggUI>().SetPlayerName(player.Player);
        playerObject.transform.GetChild(1).gameObject.GetComponent<EggUI>().SetEmptyText();
        // playerObject.transform.GetChild(0).gameObject.SetActive(false);
    }

    // Spawn ball object at specified position
    private void SpawnBall(Vector3 position, Game player)
    {
        // Spawn ball object at specified position
        GameObject ballObject = Instantiate(ballPrefab, position, Quaternion.identity, ball.transform) as GameObject;
        // ballObject.GetComponent<Renderer>().material.color = new Color(0.5f, 0.5f, 0.5f);
        ballObject.name = player.Player;
        ballObject.tag = "Ball";
        ballObject.transform.GetChild(0).gameObject.SetActive(false);
    }

    private void PrintOffsideLine(float x, string team)
    {
        if (team == "away_team")
        {
            offsideLineAway.SetPosition(0, new Vector3(x, 0.1f, 0));
            offsideLineAway.SetPosition(1, new Vector3(x, 0.1f, 68));
        }
        else
        {
            offsideLineHome.SetPosition(0, new Vector3(x, 0.1f, 0));
            offsideLineHome.SetPosition(1, new Vector3(x, 0.1f, 68));
        }
    }

    private void RemoveOffsideLine(string team)
    {
        if (team == "away_team")
        {
            offsideLineAway.SetPosition(0, new Vector3(0, 0, 0));
            offsideLineAway.SetPosition(1, new Vector3(0, 0, 0));
        }
        else
        {
            offsideLineHome.SetPosition(0, new Vector3(0, 0, 0));
            offsideLineHome.SetPosition(1, new Vector3(0, 0, 0));
        }
    }

    private int NormalizeFrameNr(int frameNr)
    {
        return frameNr - startFrame;
    }

    // Get the start frame number
    public int StartFrame()
    {
        return startFrame;
    }

    // Get the end frame number
    public int EndFrame()
    {
        return endFrame;
    }

    public int Period()
    {
        return period;
    }

    public string GetHomeTeamColor()
    {
        return homeTeamColor;
    }

    public string GetAwayTeamColor()
    {
        return awayTeamColor;
    }

    public PlayerData[] GetPlayerData()
    {
        return playerPositions;
    }
}
