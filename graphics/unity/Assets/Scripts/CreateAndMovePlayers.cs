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


    private TimeOverlay timeOverlay;

    public int startFrame = 10;
    public int endFrame = 20;

    private int currentFrame;
    private int currentFrameObjectsTracked;
    private int currentFrameNr;

    private Game[] playerData;
    private Game[] frameData;

    private int currentIndex = 0;
    private float timeInterval = 0.04f;
    private float playbackSpeed = 1.0f;
    [SerializeField] private bool isPlaying = false;

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
            currentFrame = 0;
            currentFrameObjectsTracked = frameData[0].ObjectsTracked;
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
            currentIndex = Mathf.Clamp(currentIndex + Mathf.RoundToInt(playbackSpeed * Time.deltaTime / timeInterval), 0, playerData.Length - 1);

            if (currentIndex < frameData.Length)
            {
                MoveObjects();
                if (timeOverlay != null)
                {
                    if (currentIndex < frameData.Length)
                        timeOverlay.Timer(frameData[currentIndex].MsSinceStart);
                }
                else
                    Debug.Log("TimeOverlay is null");
                currentFrameNr++;
                if (currentFrameNr < frameData.Length)
                    currentFrameObjectsTracked += frameData[currentFrameNr].ObjectsTracked;
            }
            else
            {
                Debug.Log("Frame has completed");
            }
        }

    }

    public void PlayPause()
    {
        isPlaying = !isPlaying;
    }

    public void StepForward()
    {
        currentFrameNr++;
        if (currentFrameNr < frameData.Length)
            currentFrameObjectsTracked = frameData[currentFrameNr].ObjectsTracked;
        MoveObjects();
    }

    public void StepBackward()
    {
        currentFrameNr--;
        if (currentFrameNr < frameData.Length)
            currentFrameObjectsTracked = frameData[currentFrameNr].ObjectsTracked;
        MoveObjects();
    }

    private void MoveObjects()
    {
        Vector3 position;
        Transform playerTransform;

        for (; currentFrame < currentFrameObjectsTracked; currentFrame++)
        {
            string playerName = playerData[currentFrame].PlayerName;
            playerTransform = GameObject.Find(playerName)?.transform;

            if (playerTransform == null)
            {
                Debug.Log("Player not found: " + playerName);
                if (playerData[currentFrame].TeamDirection == "right")
                {
                    position = new Vector3(105 - playerData[currentFrame].X, 0, 68 - playerData[currentFrame].Y);
                    SpawnPlayer(position, playerData[currentFrame], playerAwayPrefab, awayTeam);
                }
                else
                {
                    position = new Vector3(playerData[currentFrame].X, 0, playerData[currentFrame].Y);
                    SpawnPlayer(position, playerData[currentFrame], playerHomePrefab, homeTeam);
                }
                playerTransform = GameObject.Find(playerName)?.transform;
            }
            else
            {
                // Cache the player's transform for reuse
                position = playerTransform.position;

                if (playerData[currentFrame].TeamDirection == "right")
                {
                    position.x = 105 - playerData[currentFrame].X;
                    position.z = 68 - playerData[currentFrame].Y;
                }
                else if (playerData[currentFrame].TeamDirection == "left")
                {
                    position.x = playerData[currentFrame].X;
                    position.z = playerData[currentFrame].Y;
                }
                else
                {
                    position.x = playerData[currentFrame].X;
                    position.z = playerData[currentFrame].Y;
                    position.y = 0.1f;
                }
            }
            playerTransform.position = position;
        }
    }

    // void Update()
    // {
    //     if (currentFrameNr < frameData.Length)
    //     {
    //         timer += Time.deltaTime;

    //         if (timer >= interval)
    //         {
    //             timer = 0f;

    //             MoveObjects();
    //             if (timeOverlay != null)
    //             {
    //                 if (currentFrameNr < frameData.Length)
    //                     timeOverlay.Timer(frameData[currentFrameNr].MsSinceStart);
    //                 // Debug.Log("Time: " + frames[currentFrameNr].MsSinceStart);
    //             }
    //             else
    //                 Debug.Log("TimeOverlay is null");

    //             currentFrameNr++;
    //             if (currentFrameNr < frameData.Length)
    //                 currentFrameObjectsTracked += frameData[currentFrameNr].ObjectsTracked;
    //         }
    //     }
    //     else
    //     {
    //         Debug.Log("Frame has completed");
    //     }
    // }
    // void MoveObjects()
    // {
    //     Vector3 position;
    //     Transform playerTransform;

    //     for (; currentFrame < currentFrameObjectsTracked; currentFrame++)
    //     {
    //         string playerName = playerData[currentFrame].PlayerName;
    //         playerTransform = GameObject.Find(playerName)?.transform;

    //         if (playerTransform == null)
    //         {
    //             Debug.Log("Player not found: " + playerName);
    //             if (playerData[currentFrame].TeamDirection == "right")
    //             {
    //                 position = new Vector3(105 - playerData[currentFrame].X, 0, 68 - playerData[currentFrame].Y);
    //                 SpawnPlayer(position, playerData[currentFrame], playerAwayPrefab, awayTeam);
    //             }
    //             else
    //             {
    //                 position = new Vector3(playerData[currentFrame].X, 0, playerData[currentFrame].Y);
    //                 SpawnPlayer(position, playerData[currentFrame], playerHomePrefab, homeTeam);
    //             }
    //             playerTransform = GameObject.Find(playerName)?.transform;
    //         }
    //         else
    //         {
    //             // Cache the player's transform for reuse
    //             position = playerTransform.position;

    //             if (playerData[currentFrame].TeamDirection == "right")
    //             {
    //                 position.x = 105 - playerData[currentFrame].X;
    //                 position.z = 68 - playerData[currentFrame].Y;
    //             }
    //             else if (playerData[currentFrame].TeamDirection == "left")
    //             {
    //                 position.x = playerData[currentFrame].X;
    //                 position.z = playerData[currentFrame].Y;
    //             }
    //             else
    //             {
    //                 position.x = playerData[currentFrame].X;
    //                 position.z = playerData[currentFrame].Y;
    //                 position.y = 0.1f;
    //             }
    //         }
    //         playerTransform.position = position;
    //     }
    // }


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
