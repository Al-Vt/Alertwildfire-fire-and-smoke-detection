-- Cameras context
CREATE TABLE IF NOT EXISTS DimCamera (
    CameraID  VARCHAR PRIMARY KEY,
    Zone      VARCHAR,
    Type      VARCHAR
);

-- Time context
CREATE TABLE IF NOT EXISTS DimDate (
    DateID  VARCHAR PRIMARY KEY,
    Hour    INT,
    Season  VARCHAR
);

-- One row = one fire detection
CREATE TABLE IF NOT EXISTS FactDetection (
    CameraID    VARCHAR REFERENCES DimCamera(CameraID),
    DateID      VARCHAR REFERENCES DimDate(DateID),
    Confidence  FLOAT,
    Classe      VARCHAR
);
