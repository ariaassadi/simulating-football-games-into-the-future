using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// The Tools namespace includes all classes that are responsible for the tools in the game.
/// </summary>
namespace Tools
{
    /// <summary>
    /// The ToolManager class is responsible for managing the tools in the game.
    /// </summary>
    public class ToolManager : MonoBehaviour
    {
        private Tool[] activeTools;

        public void Update()
        {
            if (activeTools != null)
            {
                foreach (Tool tool in activeTools)
                {
                    if (!tool.IsSynchronized)
                    {
                        tool.UpdateTool();
                    }
                }
            }
        }

        /// <summary>
        /// Update the synchronized tools.
        /// </summary>
        public void UpdateSynchronized()
        {
            if (activeTools != null)
            {
                foreach (Tool tool in activeTools)
                {
                    if (tool.IsSynchronized)
                    {
                        tool.UpdateTool();
                    }
                }
            }
        }

        /// <summary>
        /// Handles what to do with the tool.
        /// </summary>
        /// <param name="toolObject">The GameObject that contains the tool.</param>
        public void SelectTool(GameObject toolObject)
        {
            Tool tool = toolObject.GetComponent<Tool>();
            if (tool != null)
            {
                if (tool.IsUniqueTool)
                {
                    if (ToolInActiveTools(tool))
                    {
                        DeselectTool(tool);
                    }
                    else
                    {
                        DeselectTools();
                        SelectToolHelper(tool);
                    }
                }
                else if (ToolInActiveTools(tool))
                {
                    DeselectTool(tool);
                }
                else
                {
                    SelectToolHelper(tool);
                }
            }
        }


        /// <summary>
        /// Add a tool to the selected tools array.
        /// </summary>
        /// <param name="tool"></param>
        private void AddToSelectedTools(Tool tool)
        {
            if (activeTools == null)
            {
                activeTools = new Tool[1];
                activeTools[0] = tool;
            }
            else
            {
                Tool[] newActiveTools = new Tool[activeTools.Length + 1];
                for (int i = 0; i < activeTools.Length; i++)
                {
                    newActiveTools[i] = activeTools[i];
                }
                newActiveTools[activeTools.Length] = tool;
                activeTools = newActiveTools;
            }
        }

        /// <summary>
        /// Check if the tool is in the active tools.
        /// </summary>
        /// <param name="tool"></param>
        /// <returns></returns>
        private bool ToolInActiveTools(Tool tool)
        {
            if (activeTools != null)
            {
                foreach (Tool activeTool in activeTools)
                {
                    if (activeTool == tool)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        /// <summary>
        /// Select a tool helper.
        /// </summary>
        /// <param name="tool"></param>
        private void SelectToolHelper(Tool tool)
        {
            if (activeTools != null)
            {
                foreach (Tool activeTool in activeTools)
                {
                    if (activeTool.IsUniqueTool)
                    {
                        DeselectTool(activeTool);
                    }
                }
            }
            tool.Select();
            AddToSelectedTools(tool);
        }

        /// <summary>
        /// Deslect a tool.
        /// </summary>
        /// <param name="tool"></param>
        private void DeselectTool(Tool tool)
        {
            if (activeTools != null)
            {
                Tool[] newActiveTools = new Tool[activeTools.Length - 1];
                int j = 0;
                for (int i = 0; i < activeTools.Length; i++)
                {
                    if (activeTools[i] != tool)
                    {
                        newActiveTools[j] = activeTools[i];
                        j++;
                    }
                    else
                    {
                        activeTools[i].Deselect();
                    }
                }
                activeTools = newActiveTools;
            }
        }

        /// <summary>
        /// Deselect all tools.
        /// </summary>
        public void DeselectTools()
        {
            if (activeTools != null)
            {
                foreach (Tool tool in activeTools)
                {
                    tool.Deselect();
                }
                activeTools = null;
            }
        }
    }
}
