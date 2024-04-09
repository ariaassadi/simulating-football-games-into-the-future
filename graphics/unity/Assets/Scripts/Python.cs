using System.Diagnostics;
using Unity.VisualScripting;
using UnityEngine;


public class PythonScript : MonoBehaviour
{

    public static void TestPythonScript(string json, string path)
    {
        // Start stopwatch for overall execution time
        Stopwatch overallStopwatch = new Stopwatch();
        overallStopwatch.Start();

        // Path to your Python script relative to the virtual environment directory
        string pythonScript;
        if (Application.platform == RuntimePlatform.Android && !Application.isEditor)
            pythonScript = Application.persistentDataPath + "/pitch_control_main.py";
        else
            pythonScript = Application.streamingAssetsPath + "/Python/pitch_control_main.py"; // Replace with actual path

        string pythonVersion;

        json = json.Replace("\"", "\\\"");
        if (Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor)
        {
            pythonVersion = "python";
        }
        else
            pythonVersion = "python3";

        // Make sure JSON string is properly formatted with double quotes

        UnityEngine.Debug.Log("JSON: " + json);
        // Arguments for the Python script (if any)
        string arguments = $"{json} {path}"; // Example arguments
        UnityEngine.Debug.Log("Arguments: " + $"{pythonScript} {arguments}");

        // Start the Python process within the virtual environment
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = $"{pythonVersion}", // Use bash to activate virtual environment
            Arguments = $"{pythonScript} {arguments}",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        Stopwatch pythonStopwatch = new Stopwatch();

        using (Process process = Process.Start(startInfo))
        {
            // Start stopwatch for Python script execution time
            pythonStopwatch.Start();

            // Read any output from the Python script
            string output = process.StandardOutput.ReadToEnd();

            if (!string.IsNullOrEmpty(output))
            {
                UnityEngine.Debug.Log(output);
            }
            // Read any errors from the Python script
            string error = process.StandardError.ReadToEnd();
            if (!string.IsNullOrEmpty(error))
            {
                UnityEngine.Debug.LogError(error);
            }

            // Wait for the process to finish
            // process.WaitForExit();
            pythonStopwatch.Stop();
            UnityEngine.Debug.Log("Time taken to execute Python script: " + pythonStopwatch.ElapsedMilliseconds + "ms");
        }

        // Stop overall execution stopwatch
        overallStopwatch.Stop();
        UnityEngine.Debug.Log("Total time taken: " + overallStopwatch.ElapsedMilliseconds + "ms");
    }

    public static void CreateDatabase()
    {
        // Start stopwatch for overall execution time
        Stopwatch overallStopwatch = new Stopwatch();
        overallStopwatch.Start();

        // Path to file
        string path = Application.temporaryCachePath + "/output.json";
        // Path to your virtual environment activate script

        // Path to your Python script relative to the virtual environment directory
        string pythonScript = Application.dataPath + "/hello_world.py"; // Replace with actual path
        string pythonVersion = "python3"; // Python version to use

        // Start stopwatch for Python script execution time

        // Arguments for the Python script (if any)
        string arguments = path; // Example arguments

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = $"{pythonVersion}", // Use bash to activate virtual environment
            Arguments = $"{pythonScript} {arguments}",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };
        // Start the Python process within the virtual environment
        // ProcessStartInfo startInfo = new ProcessStartInfo
        // {
        //     FileName = "bash", // Use bash to activate virtual environment
        //     Arguments = $"-c \"source {activateScript} && {pythonVersion} {pythonScript} {arguments}\"",
        //     UseShellExecute = false,
        //     RedirectStandardOutput = true,
        //     RedirectStandardError = true,
        //     CreateNoWindow = true
        // };
        Stopwatch pythonStopwatch = new Stopwatch();

        using (Process process = Process.Start(startInfo))
        {
            // Start stopwatch for Python script execution time
            pythonStopwatch.Start();

            // Read any output from the Python script
            string output = process.StandardOutput.ReadToEnd();
            if (!string.IsNullOrEmpty(output))
            {
                UnityEngine.Debug.Log(output);
            }
            // Read any errors from the Python script
            string error = process.StandardError.ReadToEnd();
            if (!string.IsNullOrEmpty(error))
            {
                UnityEngine.Debug.LogError(error);
            }

            // Wait for the process to finish
            // process.WaitForExit();
            pythonStopwatch.Stop();
            UnityEngine.Debug.Log("Time taken to execute Python script: " + pythonStopwatch.ElapsedMilliseconds + "ms");
        }

        // Stop overall execution stopwatch
        overallStopwatch.Stop();
        UnityEngine.Debug.Log("Total time taken: " + overallStopwatch.ElapsedMilliseconds + "ms");
    }
}
