using UnityEngine;
using System.Data;
using Unity.VisualScripting;

public class CreateAndMovePlayers : MonoBehaviour
{
    [SerializeField] private GameObject playerHomePrefab;
    [SerializeField] private GameObject playerAwayPrefab;
    [SerializeField] private GameObject ballPrefab;

    [SerializeField] private GameObject homeTeam;
    [SerializeField] private GameObject awayTeam;
    [SerializeField] private GameObject ball;

    [SerializeField] private Texture2D playIcon;
    [SerializeField] private Texture2D pauseIcon;

    [SerializeField] private GameObject playPauseButton;


    private TimeOverlay timeOverlay;

    public int startFrame = 10;
    public int endFrame = 20;

    private int currentFrameStartIndex;
    private int currentFrameEndIndex;
    private int currentFrameNr;

    private Game[] playerData;
    private Game[] frameData;

    // private int currentIndex = 0;

    // private float timeInterval = 0.04f;
    // private float playbackSpeed = 1.0f;
    private bool isPlaying = false;

    void Start()
    {
        timeOverlay = GetComponent<TimeOverlay>();
        // Retrieve players data
        playerData = DatabaseManager.query_db($"SELECT player, x, y, frame, team_direction FROM BP_vs_IKS WHERE frame>={startFrame} AND frame<={endFrame} AND period=1");

        // Retrieve frames data
        frameData = DatabaseManager.query_db($"SELECT frame, objects_tracked, ms_since_start FROM BP_vs_IKS WHERE frame>={startFrame} AND frame<={endFrame} AND period=1 GROUP BY frame");

        Debug.Log(playerData.Length);
        Debug.Log(frameData.Length);

        if (frameData != null && frameData.Length > 0)
        {
            // Loop through frames
            for (int i = 0; i < frameData[0].ObjectsTracked; i++)
            {
                SpawnObject(playerData[i]);
            }

            currentFrameNr = 0;
            currentFrameStartIndex = 0;
            currentFrameEndIndex = frameData[0].ObjectsTracked;
        }
        else
        {
            Debug.Log("No frames found");
        }
    }

    float timer = 0f;
    float interval = 0.04f; // 40 milliseconds in seconds


    void Update()
    {
        if (isPlaying)
        {
            Play();
        }
    }

    private void Play()
    {
        if (currentFrameNr < frameData.Length)
        {
            timer += Time.deltaTime;

            if (timer >= interval)
            {
                timer = 0f;

                MoveObjects();
                if (timeOverlay != null)
                {
                    if (currentFrameNr < frameData.Length)
                        timeOverlay.Timer(frameData[currentFrameNr].MsSinceStart);
                    // Debug.Log("Time: " + frames[currentFrameNr].MsSinceStart);
                }
                else
                    Debug.Log("TimeOverlay is null");

                currentFrameNr++;
                if (currentFrameNr < frameData.Length)
                    currentFrameEndIndex += frameData[currentFrameNr].ObjectsTracked;
            }
        }
        else
        {
            Debug.Log("Frame has completed");
        }
    }

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

    private void SetPlayFalse()
    {
        isPlaying = false;
        playPauseButton.GetComponent<UnityEngine.UI.RawImage>().texture = playIcon;
    }

    public void FastForward()
    {
        SetPlayFalse();
        Debug.Log("FastForward");
        if (currentFrameNr + 25 < frameData.Length)
            currentFrameNr += 25;
        else
            currentFrameNr = frameData.Length - 1;
        if (currentFrameNr < frameData.Length)
        {
            for (int i = 24; i >= 0; i--)
            {
                currentFrameEndIndex += frameData[currentFrameNr - i].ObjectsTracked;
            }
            currentFrameStartIndex = currentFrameEndIndex - frameData[currentFrameNr].ObjectsTracked;
            timeOverlay.Timer(frameData[currentFrameNr].MsSinceStart);
        }
        currentFrameEndIndex += frameData[currentFrameNr].ObjectsTracked;
        MoveObjects();
    }

    public void FastBackward()
    {
        SetPlayFalse();
        Debug.Log("FastBackward");
        if ((currentFrameNr - 25) > 0)
        {

            currentFrameNr -= 25;
            if (currentFrameNr < frameData.Length)
            {
                for (int i = 0; i < 25; i++)
                {
                    currentFrameEndIndex -= frameData[currentFrameNr + i].ObjectsTracked;
                }
                currentFrameStartIndex = currentFrameEndIndex - frameData[currentFrameNr].ObjectsTracked;
            }
            timeOverlay.Timer(frameData[currentFrameNr].MsSinceStart);
        }
        else
        {
            currentFrameNr = 0;
            currentFrameStartIndex = 0;
            currentFrameEndIndex = frameData[0].ObjectsTracked;
            timeOverlay.Timer(frameData[currentFrameNr].MsSinceStart);
        }

        Debug.Log("Current frame start: " + currentFrameStartIndex);
        Debug.Log("Current frame end: " + currentFrameEndIndex);
        MoveObjects();
    }
    public void StepForward()
    {
        SetPlayFalse();
        Debug.Log("StepForward");
        currentFrameNr++;
        if (currentFrameNr < frameData.Length)
        {
            // currentFrameStartIndex = currentFrameEndIndex;
            currentFrameEndIndex += frameData[currentFrameNr].ObjectsTracked;
            timeOverlay.Timer(frameData[currentFrameNr].MsSinceStart);
        }
        MoveObjects();
    }

    public void StepBackward()
    {
        SetPlayFalse();
        Debug.Log("StepBackward");
        currentFrameNr--;
        if (currentFrameNr < frameData.Length && currentFrameNr > 0)
        {
            currentFrameEndIndex -= frameData[currentFrameNr + 1].ObjectsTracked;
            currentFrameStartIndex = currentFrameEndIndex - frameData[currentFrameNr].ObjectsTracked;
            timeOverlay.Timer(frameData[currentFrameNr].MsSinceStart);
        }
        else
        {
            currentFrameNr = 0;
            currentFrameStartIndex = 0;
            currentFrameEndIndex = frameData[0].ObjectsTracked;
            timeOverlay.Timer(frameData[currentFrameNr].MsSinceStart);
        }
        MoveObjects();
    }

    void MoveObjects()
    {
        Vector3 position;
        Transform playerTransform;
        Debug.Log("Current frame: " + currentFrameStartIndex);

        for (int currentFrameIndex = currentFrameStartIndex; currentFrameIndex < currentFrameEndIndex; currentFrameIndex++)
        {
            string playerName = playerData[currentFrameIndex].PlayerName;
            playerTransform = GameObject.Find(playerName)?.transform;

            if (playerTransform == null)
            {
                Debug.Log("Player not found: " + playerName);
                if (playerData[currentFrameIndex].TeamDirection == "right")
                {
                    position = new Vector3(105 - playerData[currentFrameIndex].X, 0, 68 - playerData[currentFrameIndex].Y);
                    SpawnPlayer(position, playerData[currentFrameIndex], playerAwayPrefab, awayTeam);
                }
                else
                {
                    position = new Vector3(playerData[currentFrameIndex].X, 0, playerData[currentFrameIndex].Y);
                    SpawnPlayer(position, playerData[currentFrameIndex], playerHomePrefab, homeTeam);
                }
                playerTransform = GameObject.Find(playerName)?.transform;
            }
            else
            {
                // Cache the player's transform for reuse
                position = playerTransform.position;

                if (playerData[currentFrameIndex].TeamDirection == "right")
                {
                    position.x = 105 - playerData[currentFrameIndex].X;
                    position.z = 68 - playerData[currentFrameIndex].Y;
                }
                else if (playerData[currentFrameIndex].TeamDirection == "left")
                {
                    position.x = playerData[currentFrameIndex].X;
                    position.z = playerData[currentFrameIndex].Y;
                }
                else
                {
                    position.x = playerData[currentFrameIndex].X;
                    position.z = playerData[currentFrameIndex].Y;
                    position.y = 0.1f;
                }
            }
            playerTransform.position = position;
        }
    }


    void SpawnObject(Game player)
    {
        Vector3 position;
        if (player.TeamDirection == "right")
        {
            position = new Vector3(105 - player.X, 0, 68 - player.Y);
            SpawnPlayer(position, player, playerAwayPrefab, awayTeam);
        }
        else if (player.TeamDirection == "left")
        {
            position = new Vector3(player.X, 0, player.Y);
            SpawnPlayer(position, player, playerHomePrefab, homeTeam);
        }
        else
        {
            position = new Vector3(player.X, 0.1f, player.Y);
            SpawnBall(position, player);
        }
    }
    void SpawnPlayer(Vector3 position, Game player, GameObject playerPrefab, GameObject team)
    {

        GameObject playerObject = Instantiate(playerPrefab, position, Quaternion.identity, team.transform) as GameObject;
        playerObject.name = player.PlayerName;
    }

    void SpawnBall(Vector3 position, Game player)
    {
        GameObject ballObject = Instantiate(ballPrefab, position, Quaternion.identity, ball.transform) as GameObject;
        ballObject.name = player.PlayerName;
    }
}
