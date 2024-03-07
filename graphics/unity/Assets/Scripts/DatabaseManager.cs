using UnityEngine;
using System.Data; // 1
using SQLite; // 1

[Table("games")]
public class Game
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

    [Column("offside")]
    public float Offside { get; set; }

    [Column("match_id")]
    public string MatchId { get; set; }

    [Column("orientation")]
    public float Orientation { get; set; }
}

[Table("schedule")]
public class Schedule
{
    [Column("home_team_name")]
    public string HomeTeamName { get; set; }

    [Column("away_team_name")]
    public string AwayTeamName { get; set; }

    [Column("home_team_name_short")]
    public string HomeTeamNameShort { get; set; }

    [Column("away_team_name_short")]
    public string AwayTeamNameShort { get; set; }

    [Column("match_id")]
    public string MatchId { get; set; }
}


public class DatabaseManager : MonoBehaviour
{
    // QUERIES
    // public static Game[] query_games_db(int startFrame, int endFrame, int period, string match_id)
    // {
    //     // Open connection to the database
    //     var DataSource = "/home/oskarrick/uni/exjobb/simulating-football-games-into-the-future/graphics/data_processing/data/2sec.sqlite";
    //     var options = new SQLiteConnectionString(DataSource, false);
    //     var conn = new SQLiteConnection(options);
    //     // var db = new SQLiteConnection($"{Application.dataPath}/DB/bp_vs_iks.sqlite");

    //     // var query = from game in db.Table<Game>()
    //     //             where game.Frame >= startFrame &&
    //     //                   game.Frame <= endFrame &&
    //     //                   game.Period == period &&
    //     //                   game.MatchId == match_id
    //     //             select player, x, y, frame, team_direction, orientation;
    //     // string query = $"SELECT player, x, y, frame, team_direction, orientation FROM games_table_orientation WHERE frame>={startFrame} AND frame<={endFrame} AND period=1 AND match_id={match_id}";

    //     var query = conn.Table<Game>()
    //             .Where(g => g.Frame >= startFrame && g.Frame <= endFrame && g.Period == period && g.MatchId == match_id)
    //             .Select(g => new { g.Player, g.X, g.Frame, g.TeamDirection, g.Orientation })
    //             .ToList();

    //     Game[] frames = query.ToArray();
    //     db.Close();
    //     return frames;
    // }

    public static Game[] query_db(string query)
    {
        // Open connection to the database
        string conn = "/home/oskarrick/uni/exjobb/simulating-football-games-into-the-future/graphics/data_processing/data/2sec.sqlite";
        var db = new SQLiteConnection(conn);
        // var db = new SQLiteConnection($"{Application.dataPath}/DB/bp_vs_iks.sqlite");

        Game[] frames = db.Query<Game>(query).ToArray();
        db.Close();
        return frames;
    }


    public static Schedule[] query_schedule_db(string query)
    {
        string conn = "/home/oskarrick/uni/exjobb/simulating-football-games-into-the-future/graphics/data_processing/data/2sec.sqlite";
        var db = new SQLiteConnection(conn);
        // var db = new SQLiteConnection($"{Application.dataPath}/DB/bp_vs_iks.sqlite");
        Schedule[] frames = db.Query<Schedule>(query).ToArray();
        db.Close();
        return frames;
    }
}