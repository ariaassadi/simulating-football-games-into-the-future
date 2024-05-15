using UnityEngine;
using System.Data; // 1
using SQLite; // 1

[Table("games")]
public class Games
{
    [PrimaryKey, AutoIncrement]
    public int Index { get; set; }

    [Column("team")]
    public string Team { get; set; }

    [Column("team_name")]
    public string TeamName { get; set; }

    [Column("team_direction")]
    public string TeamDirection { get; set; }

    [Column("jersey_number")]
    public int JerseyNumber { get; set; }

    [Column("player")]
    public string Player { get; set; }

    [Column("x")]
    public float X { get; set; }

    [Column("y")]
    public float Y { get; set; }

    [Column("frame")]
    public int Frame { get; set; }

    [Column("period")]
    public int Period { get; set; }

    [Column("objects_tracked")]
    public int ObjectsTracked { get; set; }

    [Column("v_x")]
    public float V_X { get; set; }

    [Column("v_y")]
    public float V_Y { get; set; }

    [Column("x_future")]
    public float X_Future { get; set; }

    [Column("y_future")]
    public float Y_Future { get; set; }

    // [Column("offside")]
    // public float Offside { get; set; }

    [Column("match_id")]
    public string MatchId { get; set; }

    [Column("orientation")]
    public float Orientation { get; set; }

    public override string ToString()
    {
        return $"Player: {Player}, Team: {Team}, Jersey Number: {JerseyNumber}, Position: ({X}, {Y}), Future Position: ({X_Future}, {Y_Future})";
    }
}

[Table("schedule")]
public class GameInfo
{
    [Column("home_team_name")]
    public string HomeTeamName { get; set; }

    [Column("away_team_name")]
    public string AwayTeamName { get; set; }

    [Column("home_team_name_short")]
    public string HomeTeamNameShort { get; set; }

    [Column("away_team_name_short")]
    public string AwayTeamNameShort { get; set; }

    [Column("home_team_color")]
    public string HomeTeamColor { get; set; }

    [Column("away_team_color")]
    public string AwayTeamColor { get; set; }

    [Column("match_id")]
    public string MatchId { get; set; }

    [Column("clip")]
    public string Clip { get; set; }
}


public class DatabaseManager : MonoBehaviour
{
    // QUERIES
    public static Game[] query_db(string conn, string query)
    {
        // Open connection to the database
        var db = new SQLiteConnection(conn);
        // var db = new SQLiteConnection($"{Application.dataPath}/DB/bp_vs_iks.sqlite");

        Game[] frames = db.Query<Game>(query).ToArray();
        db.Close();
        return frames;
    }


    public static GameInfo[] query_games_db(string conn, string query)
    {
        Debug.Log(conn);

        var db = new SQLiteConnection(conn);
        // var db = new SQLiteConnection($"{Application.dataPath}/DB/bp_vs_iks.sqlite");
        GameInfo[] frames = db.Query<GameInfo>(query).ToArray();
        db.Close();
        return frames;
    }
}