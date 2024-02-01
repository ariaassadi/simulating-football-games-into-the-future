#if !DIDIMO_CORE_ENABLED
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditor.PackageManager;
using UnityEditor.PackageManager.Requests;

namespace Didimo.StorePackage.MaleBodies
{
    public static class DependencyInstaller
    {
        public static AddRequest Request;
        public readonly static string recommendedVersion = "5.0.1";
        
        //Uncomment the line below if you want to export this package.
        [UnityEditor.Callbacks.DidReloadScripts]
        public static void ResolveDependencies()
        {
            if (EditorUtility.DisplayDialog("Download and install the Didimo SDK?", 
                "In order to experience the didimos in this package, you will need to install the Didimo SDK, continue?\n\n NOTE: We recommend closing any IDEs you may have open (Visual Studio, Visual Studio Code, etc)", "OK"))
            {
                Debug.Log("Updating Didimo Core Package");
                AddPackage();
            }
        }

        public static void RequestProgress()
        {
            EditorUtility.DisplayProgressBar("Didimo SDK Core Package", "Downloading and Installing", 100f);
            if (Request.IsCompleted)
            {
                EditorApplication.update -= RequestProgress;

                EditorUtility.ClearProgressBar();
                if (Request.Status == StatusCode.Success)
                {
                    Debug.Log("Installed: " + Request.Result.packageId);
                }
                else if (Request.Status >= StatusCode.Failure)
                {
                    Debug.Log(Request.Error.message);
                    if(EditorUtility.DisplayDialog("The Didimo SDK was not installed successfuly", "Try again?\n\nNOTE: We recommend closing any IDEs you may have open (Visual Studio, Visual Studio Code, etc)", "Yes", "No"))
                    {
                        AddPackage();
                    }
                    else
                    {
                        Debug.LogError($"This package needs the Didimo Core SDK to properly work, please download it from https://github.com/didimoinc/didimo-digital-human-unity-sdk/tree/{recommendedVersion}");
                    }
                }
            }
        }

        public static void AddPackage()
        {
            Request = Client.Add($"https://github.com/didimoinc/didimo-digital-human-unity-sdk.git?path=/com.didimo.sdk.core#{recommendedVersion}");
            EditorApplication.update += RequestProgress;
        }
    }
}
#endif