using System.Diagnostics;
using Unity.VisualScripting;
using UnityEngine;
using System;
using System.Net.Sockets;
using System.Text;


public class PythonScript : MonoBehaviour
{
    private static Process _pythonProcess;
    private static Socket _pythonSocket;

    private static byte[] _receiveBuffer = new byte[262144]; // Adjust buffer size as needed

    public static bool StartPitchControlScript(string path, int timeoutMilliseconds = 10000)
    {
        if (_pythonProcess != null && !_pythonProcess.HasExited)
        {
            UnityEngine.Debug.LogError("Python script is already running.");
            return false;
        }

        // Python version to use based on the platform
        string pythonVersion;
        if (Application.platform == RuntimePlatform.WindowsPlayer || Application.platform == RuntimePlatform.WindowsEditor)
        {
            pythonVersion = "python";
        }
        else
            pythonVersion = "python3";

        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = $"{pythonVersion}",
            Arguments = $"{path}",
            UseShellExecute = false,
            RedirectStandardInput = false,
            RedirectStandardOutput = false,
            RedirectStandardError = false,
            CreateNoWindow = true
        };

        _pythonProcess = Process.Start(startInfo);

        if (_pythonProcess == null)
        {
            UnityEngine.Debug.LogError("Failed to start Python script.");
            return false;
        }

        UnityEngine.Debug.Log("Python script started.");
        return true;
    }


    public static bool ConnectToPitchControlScript(string host = "localhost", int port = 12345, int timeoutMilliseconds = 5000)
    {
        _pythonSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);

        DateTime startTime = DateTime.Now;

        try
        {
            while (!_pythonSocket.Connected)
            {
                try
                {
                    _pythonSocket.Connect(host, port);
                    UnityEngine.Debug.Log("Connected to Python script.");
                    return true; // Connection successful
                }
                catch (SocketException)
                {
                    // Connection failed, wait a short time before retrying
                    System.Threading.Thread.Sleep(100);
                }

                // Check if timeout has been reached
                if ((DateTime.Now - startTime).TotalMilliseconds > timeoutMilliseconds)
                {
                    UnityEngine.Debug.LogError("Connection timeout reached.");
                    return false; // Connection timeout
                }
            }

            return true; // Connection successful
        }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogError($"Failed to connect to Python script: {ex.Message}");
            return false; // Connection failed due to an exception
        }
    }

    public static bool CloseConnectionToPitchControlScript()
    {
        try
        {
            if (_pythonSocket != null && _pythonSocket.Connected)
            {
                _pythonSocket.Shutdown(SocketShutdown.Both); // Shutdown both send and receive operations
                _pythonSocket.Close(); // Close the socket
                _pythonSocket = null;
                UnityEngine.Debug.Log("Connection to Python script closed.");
                return true;
            }
            else
            {
                UnityEngine.Debug.LogWarning("Not connected to Python script.");
                return false;
            }
        }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogError($"Error while closing connection to Python script: {ex.Message}");
            return false;
        }
    }



    public static string SendDataToPitchControlScript(string jsonData)
    {
        if (_pythonSocket == null || !_pythonSocket.Connected)
        {
            UnityEngine.Debug.LogError("Not connected to Python script.");
            return null;
        }

        try
        {
            byte[] data = Encoding.UTF8.GetBytes(jsonData);
            if (data.Length > 8192)
            {
                UnityEngine.Debug.LogWarning("Data size: " + data.Length);
                UnityEngine.Debug.LogWarning("Data: " + jsonData);
                UnityEngine.Debug.LogError("Data size exceeds 8192 bytes. Consider splitting the data into smaller chunks.");
            }
            UnityEngine.Debug.Log("Data size: " + data.Length);
            _pythonSocket.Send(data);
            UnityEngine.Debug.Log("Data sent to Python script.");
        }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogError($"Failed to send data to Python script: {ex.Message}");
        }

        string recvJson = ReceiveDataFromPitchControlScript(5000);
        if (recvJson == null)
        {
            UnityEngine.Debug.LogWarning("Failed to receive data from Python script.\nData sent: " + jsonData);
            return null;
        }
        UnityEngine.Debug.Log("Received data: " + recvJson);
        return recvJson;
    }

    public static string ReceiveDataFromPitchControlScript(int timeoutMilliseconds)
    {
        if (_pythonSocket == null || !_pythonSocket.Connected)
        {
            UnityEngine.Debug.LogError("Not connected to Python script.");
            return null;
        }

        try
        {
            DateTime startTime = DateTime.Now;
            StringBuilder receivedData = new StringBuilder();

            while (true)
            {
                if (_pythonSocket.Available > 0)
                {
                    int bytesRead = _pythonSocket.Receive(_receiveBuffer);
                    if (bytesRead > 0)
                    {
                        receivedData.Append(Encoding.UTF8.GetString(_receiveBuffer, 0, bytesRead));
                        // UnityEngine.Debug.Log("Data received from Python script: " + receivedData);
                        // Check if the received data is complete (e.g., ends with a specific delimiter)
                        if (receivedData.ToString().EndsWith("}"))
                        {
                            return receivedData.ToString();
                        }
                    }
                }

                // Check if the timeout has been reached
                if ((DateTime.Now - startTime).TotalMilliseconds > timeoutMilliseconds)
                {
                    UnityEngine.Debug.LogWarning("Timeout reached while waiting for response from Python script.");
                    return null;
                }

                // Add a small delay to avoid busy-waiting
                System.Threading.Thread.Sleep(10);
            }
        }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogError($"Failed to receive data from Python script: {ex.Message}");
            return null;
        }
    }


    public static bool StopPitchControlScript()
    {
        try
        {
            if (_pythonProcess != null && !_pythonProcess.HasExited)
            {
                // Send a termination signal to the Python process
                _pythonProcess.Kill();

                // Wait for the process to exit
                _pythonProcess.WaitForExit();

                // Close the process
                _pythonProcess.Close();

                // Reset the process reference
                _pythonProcess = null;

                UnityEngine.Debug.Log("Python script stopped.");
                return true;
            }
            else
            {
                UnityEngine.Debug.LogWarning("Python script is not running.");
                return true;
            }
        }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogError($"Error while stopping Python script: {ex.Message}");
            return false;
        }
    }

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
