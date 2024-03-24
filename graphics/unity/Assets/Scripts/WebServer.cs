// using UnityEngine;
// using UnityEngine.Networking;
// using System.Collections;

// // UnityWebRequest to send and receive data from a webserver
// // https://docs.unity3d.com/Manual/UnityWebRequest.html

// public class WebServer : MonoBehaviour
// {
//     // Send a JSON to the webserver to get the pitch control data
//     public static string GetPitchControlJSON(string json)
//     {
//         return StartCoroutine(Upload(json));
//     }


//     IEnumerator Upload(string json)
//     {
//         using (UnityWebRequest www = UnityWebRequest.Post("https://www.my-server.com/myapi", json, "application/json"))
//         {
//             yield return www.SendWebRequest();

//             if (www.result != UnityWebRequest.Result.Success)
//             {
//                 Debug.LogError(www.error);
//             }
//             else
//             {
//                 Debug.Log("Form upload complete!");
//                 // Print the response
//                 Debug.Log(www.downloadHandler.text);
//                 // return the response
//                 return www.downloadHandler.text;
//             }
//         }
//     }

// }