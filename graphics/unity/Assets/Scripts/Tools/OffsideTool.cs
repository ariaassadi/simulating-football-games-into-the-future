using UnityEngine;

using GameVisualization;

namespace Tools
{

    public class OffsideTool : Tool
    {
        private GameManager gameManager;

        private GameInfo gameInfo;

        private LineRenderer offsideLineHome;
        private LineRenderer offsideLineAway;



        private float homeOffsideLine = 0;
        private float awayOffsideLine = 0;
        public override void UpdateTool()
        {
            UpdateOffsideLines();
        }

        protected override void Start()
        {
            base.Start();
            synchronized = true;
            gameManager = GameObject.Find("GameManager").GetComponent<GameManager>();
        }

        public override void Select()
        {
            base.Select();
            // Create offside lines
            InstantiateOffsideLines();
            Debug.Log("Offside tool selected");
        }

        public override void Deselect()
        {
            base.Deselect();
            // Destroy offside lines
            DestroyOffsideLines();
            Debug.Log("Offside tool deselected");
        }

        private void UpdateOffsideLines()
        {
            PlayerData[] playerData = gameManager.GetPlayerData();
            gameInfo = gameManager.GetGameInfo();

            foreach (PlayerData player in playerData)
            {
                if (player.offside != 0)
                {
                    if (player.team == "home_team")
                    {
                        Debug.Log("Home team offside line: " + player.offside);
                        homeOffsideLine = player.offside;
                    }
                    else if (player.team == "away_team")
                    {
                        Debug.Log("Away team offside line: " + player.offside);
                        awayOffsideLine = player.offside;
                    }
                    else
                    {
                        Debug.Log("Error: Not offside for ball");
                    }
                }
            }

            if (homeOffsideLine != 0)
            {
                // Debug.Log("Home team is offside: " + homeOffsideLine);
                PrintOffsideLine(homeOffsideLine, "home_team");
            }
            else
            {
                // Debug.Log("Home team not offside: " + homeOffsideLine);
                RemoveOffsideLine("home_team");
            }
            if (awayOffsideLine != 0)
            {
                // Debug.Log("Away team is offside: " + awayOffsideLine);
                PrintOffsideLine(awayOffsideLine, "away_team");
            }
            else
            {
                // Debug.Log("Away team not offside: " + awayOffsideLine);
                RemoveOffsideLine("away_team");
            }

            // Reset the offside line values
            homeOffsideLine = 0;
            awayOffsideLine = 0;
        }

        private void InstantiateOffsideLines()
        {
            GameObject offsideLineHomeG = new GameObject("Offside Line Home", typeof(LineRenderer));
            GameObject offsideLineAwayG = new GameObject("Offside Line Away", typeof(LineRenderer));

            offsideLineHomeG.transform.SetParent(transform);
            offsideLineAwayG.transform.SetParent(transform);

            offsideLineHome = offsideLineHomeG.GetComponent<LineRenderer>();
            offsideLineAway = offsideLineAwayG.GetComponent<LineRenderer>();

            offsideLineHome.startWidth = 0.2f;
            offsideLineHome.endWidth = 0.2f;
            offsideLineHome.positionCount = 2;
            offsideLineHome.material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
            offsideLineHome.material.color = Color.red;

            offsideLineAway.startWidth = 0.2f;
            offsideLineAway.endWidth = 0.2f;
            offsideLineAway.positionCount = 2;
            offsideLineAway.material = new Material(Shader.Find("Universal Render Pipeline/Unlit"));
            offsideLineAway.material.color = Color.red;

            UpdateOffsideLines();
        }

        private void DestroyOffsideLines()
        {
            Debug.Log("Destroying offside lines");
            GameObject offsideLineHomeG = GameObject.Find("Offside Line Home");
            GameObject offsideLineAwayG = GameObject.Find("Offside Line Away");

            Destroy(offsideLineHomeG);
            Destroy(offsideLineAwayG);

        }

        private void PrintOffsideLine(float x, string team)
        {
            if (team == "away_team")
            {
                offsideLineAway.SetPosition(0, new Vector3(x, 0.1f, 0));
                offsideLineAway.SetPosition(1, new Vector3(x, 0.1f, -68));
            }
            else
            {
                offsideLineHome.SetPosition(0, new Vector3(x, 0.1f, 0));
                offsideLineHome.SetPosition(1, new Vector3(x, 0.1f, -68));
            }
        }

        private void RemoveOffsideLine(string team)
        {
            if (team == "away_team")
            {
                offsideLineAway.SetPosition(0, new Vector3(0, 0, 0));
                offsideLineAway.SetPosition(1, new Vector3(0, 0, 0));
            }
            else
            {
                offsideLineHome.SetPosition(0, new Vector3(0, 0, 0));
                offsideLineHome.SetPosition(1, new Vector3(0, 0, 0));
            }
        }
    }

}