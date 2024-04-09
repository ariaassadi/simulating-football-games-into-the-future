using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using System.Linq;

public class Utils : MonoBehaviour
{
    /// <summary>
    /// Convert hex color to Color
    /// </summary>
    /// <param name="hex"></param>
    /// <returns>The Color</returns>
    public static Color HexToColor(string hex)
    {
        if (hex.StartsWith("#"))
        {
            hex = hex.Substring(1);
        }
        if (hex.Length != 6)
        {
            Debug.LogError("Invalid hex color");
            return Color.white;
        }
        float r = int.Parse(hex.Substring(0, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
        float g = int.Parse(hex.Substring(2, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
        float b = int.Parse(hex.Substring(4, 2), System.Globalization.NumberStyles.HexNumber) / 255f;
        return new Color(r, g, b);
    }

    /// <summary>
    /// Change opacity of a color using hex
    /// </summary>
    /// <param name="hex"></param>
    /// <param name="alpha"></param>
    /// <returns>The Color</returns>
    public static Color HexToColor(string hex, float alpha)
    {
        Color color = HexToColor(hex);
        color.a = alpha;
        return color;
    }

    public static Color ChangeOpacity(Color color, float alpha)
    {
        color.a = alpha;
        return color;
    }

    public static Color ChangeBrightness(Color color, float factor)
    {
        float h, s, v;
        Color.RGBToHSV(color, out h, out s, out v);
        v *= factor;
        return Color.HSVToRGB(h, s, v);
    }

    public static Color GenerateColorGradient(float value)
    {
        // Clamp the value between -1 and 1
        value = Mathf.Clamp(value, -1f, 1f);

        // Define the colors for red, white, and blue
        Color red = Color.red;
        Color white = Color.white;
        Color blue = Color.blue;

        // Interpolate between the colors based on the input value
        if (value < 0f)
        {
            return Color.Lerp(red, white, Mathf.InverseLerp(-1f, 0f, value));
        }
        else
        {
            return Color.Lerp(white, blue, Mathf.InverseLerp(0f, 1f, value));
        }
    }

    // GEnerateColor Gradient with custom colors
    public static Color GenerateColorGradient(float value, Color homeColor, Color awayColor)
    {
        // Clamp the value between -1 and 1
        value = Mathf.Clamp(value, -1f, 1f);

        // Interpolate between the colors based on the input value
        if (value < 0f)
        {
            return Color.Lerp(awayColor, Color.white, Mathf.InverseLerp(-1f, 0f, value));
        }
        else
        {
            return Color.Lerp(Color.white, homeColor, Mathf.InverseLerp(0f, 1f, value));
        }
    }
}


[System.Serializable]
public class Position
{
    public float x;
    public float y;
}

public static class JsonParser
{
    // Define your JSON string

    public static PositionData ParseJSON()
    {
        string jsonString = "{\"position\": [{\"x\": 0, \"y\": 1},{\"x\": 2, \"y\": 3},{\"x\": 4, \"y\": 5},{\"x\": 6, \"y\": 7}]}";
        // Parse JSON into Position array
        PositionData positionData = JsonUtility.FromJson<PositionData>(jsonString);

        // Access the positions

        // Assuming positionData is your PositionData object containing the positions

        return positionData;
    }

    // Convert JSON to Vector2 array
    public static Vector2[] JsonToVectorArray(string jsonString = "{\"position\": [{\"x\": 0, \"y\": 1},{\"x\": 2, \"y\": 3},{\"x\": 4, \"y\": 5},{\"x\": 6, \"y\": 7}]}")
    {
        PositionData positionData = JsonUtility.FromJson<PositionData>(jsonString);
        Debug.Log("Converted from JSON" + string.Join(", ", positionData.position.Select(pos => $"X: {pos.x}, Y: {pos.y}")));
        Vector2[] vectors = new Vector2[positionData.position.Length];

        for (int i = 0; i < positionData.position.Length; i++)
        {
            vectors[i] = new Vector2(positionData.position[i].x, positionData.position[i].y);
        }

        return vectors;
    }

    public static string Vector2ArrayToJson(Vector2[] vectors)
    {
        PositionData positionData = new PositionData();
        positionData.position = new Position[vectors.Length];

        for (int i = 0; i < vectors.Length; i++)
        {
            positionData.position[i] = new Position();
            positionData.position[i].x = vectors[i].x;
            positionData.position[i].y = vectors[i].y;
        }

        Debug.Log("Conveted to JSON:\n" + JsonUtility.ToJson(positionData));

        return JsonUtility.ToJson(positionData);
    }

    public static float[,] ParsePitchJSON(string json)
    {
        // Read jsonString from a file
        // string jsonString = File.ReadAllText("Assets/Static/Coordinates/pitch_control.json");

        // Parse JSON into a wrapper class
        PitchData pitchData = JsonUtility.FromJson<PitchData>(json);

        // Access the pitch values
        int rows = 68;
        int cols = 105;
        // convert pitchData to a 2D array of floats
        float[,] pitch = new float[rows, cols];
        int index = 0;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                pitch[i, j] = pitchData.pitch[index];
                index++;
            }
        }

        return pitch;
    }
    public static PositionDataHA ParseJSONHA()
    {
        string jsonString = "{\"home\": [{\"x\": 0, \"y\": 1},{\"x\": 2, \"y\": 3},{\"x\": 4, \"y\": 5},{\"x\": 6, \"y\": 7}], \"away\": [{\"x\": 0, \"y\": 1},{\"x\": 2, \"y\": 3},{\"x\": 4, \"y\": 5},{\"x\": 6, \"y\": 7}], \"ball\": {\"x\": 0, \"y\": 1}}";
        // Parse JSON into Position array
        PositionDataHA positionData = JsonUtility.FromJson<PositionDataHA>(jsonString);

        // Debug log the positionData
        Debug.Log("Converted from JSON home" + string.Join(", ", positionData.home.Select(pos => $"X: {pos.x}, Y: {pos.y}")));
        Debug.Log("Converted from JSON away" + string.Join(", ", positionData.away.Select(pos => $"X: {pos.x}, Y: {pos.y}")));
        Debug.Log("Converted from JSON ball" + $"X: {positionData.ball.x}, Y: {positionData.ball.y}");


        return positionData;
    }

    // Convert an array of PlayerData to JSON
    public static string PlayerDataArrayToJson(PlayerData[] playerDataArray)
    {
        PlayerDataWrapper wrapper = new PlayerDataWrapper();
        wrapper.players = playerDataArray;
        string jsonString = JsonUtility.ToJson(wrapper);
        Debug.Log("Converted to JSON: " + jsonString);
        return jsonString;
    }

    public static string PlayerDataToJson()
    {
        PlayerData playerData = new PlayerData();
        playerData.x = 0;
        playerData.y = 1;
        playerData.team = "home";
        playerData.jersey_number = 10;

        return JsonUtility.ToJson(playerData);
    }

    public static PlayerData ParsePlayerData()
    {
        string jsonString = "{\"x\": \"0\", \"y\": \"1\", \"team\": \"home\", \"jersey_number\": 10}";
        // Parse JSON into Position array
        PlayerData playerData = JsonUtility.FromJson<PlayerData>(jsonString);

        // Debug log the positionData
        Debug.Log("Converted from JSON player" + $"X: {playerData.x}, Y: {playerData.y}, Team: {playerData.team}, Jersey Number: {playerData.jersey_number}");

        return playerData;
    }

}

[System.Serializable]
public class PositionData
{
    public Position[] position;
}

[System.Serializable]
public class PositionDataHA
{
    public Position[] home;
    public Position[] away;
    public Position ball;
}

[System.Serializable]
public class PitchData
{
    public float[] pitch;
}

[System.Serializable]
public class PlayerData
{
    public float x;
    public float y;
    public string team;
    public int jersey_number;

    public float v;
    public float orientation;
}

[System.Serializable]
public class PlayerDataWrapper
{
    public PlayerData[] players;
}


// [System.Serializable]
// public class Coordinates
// {
//     public float x;
//     public float y;
//     public override string ToString()
//     {
//         return $"x={x.ToString()}, z={y.ToString()}";
//     }
// }


// public class JSONParser
// {
//     [SerializeField] public TextAsset file;

//     // public static Coordinates[] ParseJSONFile(string filePath)
//     // {
//     //     using (StreamReader streamReader = new StreamReader(filePath))
//     //     {
//     //         string jsonString = streamReader.ReadToEnd();
//     //         // Debug.Log(jsonString);
//     //         Coordinates[] positions = JsonConvert.DeserializeObject<Coordinates>(jsonString);
//     //         // Debug.Log("PLAYA: " + matchData.players[0].player_name);
//     //         // Debug.Log("PLAYA CO x: " + matchData.players[0].coordinates[0][0].ToString());
//     //         // Debug.Log("PLAYA CO: " + matchData.players[0].coordinates.ToString());

//     //         return positions;
//     //     }
//     // }

//     public static Coordinates[] ParseJSONFile(string filePath)
//     {
//         using (StreamReader streamReader = new StreamReader(filePath))
//         {
//             string jsonString = streamReader.ReadToEnd();
//             Debug.Log(jsonString);
//             Coordinates[] positions = JsonHelper.FromJson<Coordinates>(jsonString);
//             // Debug.Log("PLAYA: " + matchData.players[0].player_name);
//             // Debug.Log("PLAYA CO x: " + matchData.players[0].coordinates[0][0].ToString());
//             // Debug.Log("PLAYA CO: " + matchData.players[0].coordinates.ToString());
//             if (positions == null)
//             {
//                 Debug.Log("No positions found");
//                 return null;
//             }
//             foreach (Coordinates pos in positions)
//             {
//                 Debug.Log("POS: " + pos.ToString());
//             }
//             return positions;
//         }
//     }
// }

// public static class JsonHelper
// {
//     public static T[] FromJson<T>(string json)
//     {
//         Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(json);
//         return wrapper.Items;
//     }

//     public static string ToJson<T>(T[] array)
//     {
//         Wrapper<T> wrapper = new Wrapper<T>();
//         wrapper.Items = array;
//         return JsonUtility.ToJson(wrapper);
//     }

//     public static string ToJson<T>(T[] array, bool prettyPrint)
//     {
//         Wrapper<T> wrapper = new Wrapper<T>();
//         wrapper.Items = array;
//         return JsonUtility.ToJson(wrapper, prettyPrint);
//     }

//     [System.Serializable]
//     private class Wrapper<T>
//     {
//         public T[] Items;
//     }
// }