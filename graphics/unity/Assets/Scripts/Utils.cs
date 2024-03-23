using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;


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

    public static Color ChangeBrightness(Color color, float factor)
    {
        float h, s, v;
        Color.RGBToHSV(color, out h, out s, out v);
        v *= factor;
        return Color.HSVToRGB(h, s, v);
    }
}

[System.Serializable]
public class Coordinates
{
    public float x;
    public float z;
    public override string ToString()
    {
        return $"x={x.ToString()}, z={z.ToString()}";
    }
}


// public class JSONParser
// {
//     [SerializeField] public TextAsset file;

//     public static Coordinates ParseJSONFile(string filePath)
//     {
//         using (StreamReader streamReader = new StreamReader(filePath))
//         {
//             string jsonString = streamReader.ReadToEnd();
//             // Debug.Log(jsonString);
//             MatchData matchData = JsonConvert.DeserializeObject<MatchData>(jsonString);
//             // Debug.Log("PLAYA: " + matchData.players[0].player_name);
//             // Debug.Log("PLAYA CO x: " + matchData.players[0].coordinates[0][0].ToString());
//             // Debug.Log("PLAYA CO: " + matchData.players[0].coordinates.ToString());

//             return matchData;
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