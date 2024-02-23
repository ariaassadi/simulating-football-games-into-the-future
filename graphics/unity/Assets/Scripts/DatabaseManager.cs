using UnityEngine;
using System.Data; // 1
using SQLite; // 1

[Table("BP_vs_IKS")]
public class Game
{
    [Column("id")]
    public string Team
    { get; set; }

    [Column("team_name")]
    public string TeamName
    { get; set; }

    [Column("team_direction")]
    public string TeamDirection
    { get; set; }

    [Column("jersey_number")]
    public string JerseyNumber
    { get; set; }

    [Column("player")]
    public string PlayerName
    { get; set; }

    [Column("role")]
    public string Role
    { get; set; }

    [Column("distance_ran")]
    public float DistanceRan
    { get; set; }

    [Column("x")]
    public float X
    { get; set; }

    [Column("y")]
    public float Y
    { get; set; }

    [Column("frame")]
    public int Frame
    { get; set; }

    [Column("minute")]
    public int Minute
    { get; set; }

    [Column("second")]
    public int Second
    { get; set; }

    [Column("period")]
    public int Period
    { get; set; }

    [Column("ms_since_start")]
    public int MsSinceStart
    { get; set; }

    [Column("events")]
    public string Events
    { get; set; }

    [Column("objects_tracked")]
    public int ObjectsTracked
    { get; set; }
}
public class DatabaseManager : MonoBehaviour
{
    // QUERIES
    public static Game[] query_db(string query)
    {
        var db = new SQLiteConnection($"{Application.dataPath}/DB/bp_vs_iks.sqlite");
        Game[] frames = db.Query<Game>(query).ToArray();
        db.Close();
        return frames;
    }
}