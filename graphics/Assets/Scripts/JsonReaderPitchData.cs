using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;

[System.Serializable]
public class Coordinates
{
    public float x;
    public float y;
    public float z;
    public override string ToString()
    {
        return $"x={x.ToString()}, y={y.ToString()}, z={z.ToString()}";
    }
}

[System.Serializable]
public class PlayerData
{
    public string team;
    public string player_name;
    public float[] coordinates;
}

[System.Serializable]
public class MatchData
{
    public PlayerData[] players;
}

public class JSONParser
{
    [SerializeField] public TextAsset file;

    public static MatchData ParseJSONFile(string filePath)
    {
        using (StreamReader streamReader = new StreamReader(filePath))
        {
            string jsonString = streamReader.ReadToEnd();
            Debug.Log(jsonString);
            MatchData matchData = JsonConvert.DeserializeObject<MatchData>(jsonString);
            Debug.Log("PLAYA: " + matchData.players[0].player_name);
            // Debug.Log("PLAYA CO x: " + matchData.players[0].coordinates[0][0].ToString());
            // Debug.Log("PLAYA CO: " + matchData.players[0].coordinates.ToString());

            return matchData;
        }
    }
}

public static class JsonHelper
{
    public static T[] FromJson<T>(string json)
    {
        Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(json);
        return wrapper.Items;
    }

    public static string ToJson<T>(T[] array)
    {
        Wrapper<T> wrapper = new Wrapper<T>();
        wrapper.Items = array;
        return JsonUtility.ToJson(wrapper);
    }

    public static string ToJson<T>(T[] array, bool prettyPrint)
    {
        Wrapper<T> wrapper = new Wrapper<T>();
        wrapper.Items = array;
        return JsonUtility.ToJson(wrapper, prettyPrint);
    }

    [System.Serializable]
    private class Wrapper<T>
    {
        public T[] Items;
    }
}
