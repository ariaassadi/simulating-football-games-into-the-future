using System.Diagnostics;
using UnityEditor;
using UnityEngine;

public static class PythonScript
{
    [MenuItem("Tools/Run Python Script")]
    public static void TestPythonScript()
    {
        // Start stopwatch for overall execution time
        Stopwatch overallStopwatch = new Stopwatch();
        overallStopwatch.Start();

        // Path to file
        string path = Application.temporaryCachePath + "/outpus.json";
        // Path to your virtual environment activate script
        string activateScript = "/home/oskarrick/uni/exjobb/simulating-football-games-into-the-future/venv/bin/activate"; // Replace with actual path

        // Path to your Python script relative to the virtual environment directory
        string pythonScript = Application.dataPath + "/hello_world.py"; // Replace with actual path
        string pythonVersion = "python3"; // Python version to use

        // Start stopwatch for Python script execution time

        // Arguments for the Python script (if any)
        string arguments = path; // Example arguments

        // Start the Python process within the virtual environment
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = "bash", // Use bash to activate virtual environment
            Arguments = $"-c \"source {activateScript} && {pythonVersion} {pythonScript} {arguments}\"",
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
}
