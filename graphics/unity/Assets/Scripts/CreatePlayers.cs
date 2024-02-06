using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.Universal; // Import URP namespace

public class CreatePlayers : MonoBehaviour
{
    public GameObject playerPrefab;

    // [SerializeField] public TextAsset file;

    [SerializeField] public MatchData parsedData;

    void Start()
    {
        string filePath = "Assets/Static/Coordinates/coordinates.json";
        parsedData = JSONParser.ParseJSONFile(filePath);

        // Accessing the parsed data

        PlayerData[] players = parsedData.players;
        // Printing the data for demonstration
        // Debug.Log("Players Home Team: parsed");
        // Debug.Log("Number of players: " + players.Length);
        // Debug.Log("FIRST PLAYER:" + players[0].player_name);
        // Debug.Log("FIRST PLAYER coordinate:" + players[0].coordinates[0].ToString());
        for (int i = 0; i < players.Length; i++)
        {
            Vector3 position = new Vector3(players[i].coordinates[0], players[i].coordinates[1], players[i].coordinates[2]);
            // Debug.Log("  " + players[i].player_name + " - " + position.ToString());
            SpawnPlayer(position, players[i]);
        }
    }

    void SpawnPlayer(Vector3 position, PlayerData player)
    {
        GameObject playerObject = Instantiate(playerPrefab, position, Quaternion.identity) as GameObject;
        playerObject.name = player.player_name;

    }
}

