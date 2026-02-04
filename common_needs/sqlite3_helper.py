##########################################
### Import needed general dependencies ###
##########################################
# Internal modules
from type_helper import isDictionaryWithStringKeys, isListWithStringEntries

# External modules
from os.path import exists
from pathlib import Path, PosixPath, WindowsPath
from sqlite3 import connect, Connection, Cursor
from typing import Any, Tuple, Union


####################################################
### Define a shared list of allowed column types ###
####################################################
# Initialize the list
ALLOWED_COLUMN_TYPES = []

# Add the various numeric types
ALLOWED_COLUMN_TYPES += ["INT", "INTEGER", "TINYINT", "SMALLINT", "MEDIUMINT", "BIGINT", "UNSIGNED BIG INT", "INT2", "INT8"]
ALLOWED_COLUMN_TYPES += ["NUMERIC", "DECIMAL(10, 5)", "BOOLEAN", "DATE", "DATETIME"]
ALLOWED_COLUMN_TYPES += ["REAL", "DOUBLE", "DOUBLE PRECISION", "FLOAT"]

# Add all other types
ALLOWED_COLUMN_TYPES += ["CHARACTER(20)", "VARCHAR(255)", "VARYING CHARACTER(255)", "NCHAR(55)", "NATIVE CHARACTER(70)", "NVARCHAR(100)", "TEXT", "CLOB"]
ALLOWED_COLUMN_TYPES += ["BLOB", "no datatype specified"]


###########################################################
### Define internal functions which are frequently used ###
###########################################################
def _checkDBPath(db_path:Union[PosixPath, WindowsPath]) -> Tuple[Connection, Cursor]:
	# Verify that the provided db file location is valid, returns a connection and cursor for thedb file
	# Make sure the filename in the path is valid
	assert type(db_path) in [PosixPath, WindowsPath], "_checkDBPath: Provided value for 'db_path' must be a PosixPath or WindowsPath object"
	assert db_path.name.endswith(".db") == True, "_checkDBPath: Provided value for 'db_path' must refer to a filename ending with '.db'"
	assert len(db_path.name) > 3, "_checkDBPath: Provided value for 'db_path' must refer to a filename of length > 3"
	
	# Make sure the file exists
	assert exists(db_path.parent) == True, "_checkDBPath: Provided value for 'db_path' must refer to a filename in an existing folder"
	
	# Connect to the db file and create a cursor
	db_connection = connect(db_path)
	db_cursor = db_connection.cursor()
	
	# Return the results
	return db_connection, db_cursor
	
def _checkTableName(db_cursor:Cursor, table_name:str = None, exists_flag:bool = True) -> list:
	# Verify that the table is (or isn't) present in the given db file (without checking existence of the db file), returns the list of table names
	# Make sure that the table name is valid (if needed)
	if table_name is not None:
		assert type(table_name) == str, "_checkTableName: If provided, value for 'table_name' must be a str object"
		assert len(table_name) > 0, "_checkTableName: If provided, value for 'table_name' must be a non-empty string"
	
	# Fetch all tables in the database
	table_names_query = "SELECT name FROM sqlite_master WHERE type = 'table';"
	table_names_result = db_cursor.execute(table_names_query).fetchall()
	table_names = [value[0] for value in table_names_result]
	
	# Handle the various cases (if needed)
	if table_name is not None:
		if exists_flag == True:
			# Make sure the given table exists in the database
			assert table_name in table_names, "_checkTableName: Provided value for 'table_name' doesn't correspond to an existing table in the db file (when it should)"
		else:
			# Make sure the given table doesn't exist in the database
			assert table_name not in table_names, "_checkTableName: Provided value for 'table_name' corresponds to an existing table in the db file (when it shouldn't)"
		
	# Return the results
	return table_names
		
def _checkColumnName(db_cursor:Cursor, table_name:str, column_name:str = None, exists_flag:bool = True) -> list:
	# Verify that the column name is (or isn't) present in the given table of the given db file (without checking existence of the db file or table), returns a list of column names
	# Make sure that the column name is valid (if needed)
	if column_name is not None:
		assert type(column_name) == str, "_checkColumnName: If provided, value for 'column_name' must be a str object"
		assert len(column_name) > 0, "_checkColumnName: If provided, value for 'column_name' must be a non-empty string"
	
	# Fetch all column names in the table
	column_names_query = "SELECT name FROM PRAGMA_TABLE_INFO('" + table_name + "');"
	column_names_result = db_cursor.execute(column_names_query).fetchall()
	column_names = [value[0] for value in column_names_result]
	
	# Handle the various cases (if needed)
	if column_name is not None:
		if exists_flag == True:
			# Make sure the given column exists in the table
			assert column_name in column_names, "_checkColumnName: Provided value for 'column_name' doesn't correspond to an existing column in requested table (when it should)"
		else:
			# Make sure the given column doesn't exist in the table
			assert column_name not in column_names, "_checkColumnName: Provided value for 'column_name' corresponds to an existing column in requested table (when it shouldn't)"
		
	# Return the results
	return column_names
	
def _checkRowCount(db_cursor:Cursor, table_name:str, min_count:int = None) -> int:
	# Verify that the row count in the given table of the given db file is above the threshold (without checking existence of the db file or table), returns the number of rows
	# Make sure that the minimum row count is valid (if needed)
	if min_count is not None:
		assert type(min_count) == int, "_checkRowCount: If provided, value for 'row_count' must be an int object"
		assert min_count >= 0, "_checkRowCount: If provided, value for 'row_count' must be non-negative"
		
	# Fetch the number of rows from this table
	row_count_query = "SELECT COUNT(*) FROM '" + table_name + "';"
	row_count_result = db_cursor.execute(row_count_query).fetchone()
	row_count = row_count_result[0]
	
	# Check that the row count is large enough (if needed)
	if min_count is not None:
		assert row_count >= min_count, "_checkRowCount: Relevant table doesn't have a row count >= the provided threshold"
		
	# Return the results
	return row_count


##########################################################################################
### Define functions for adding and deleting tables in a db file at the given location ###
##########################################################################################
def addTable(db_path:Union[PosixPath, WindowsPath], table_name:str, column_names:list, column_types:list, replace_flag:bool = True):
	# Add a table to the given db file (or replaces an existing table with an empty one if needed)
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	if replace_flag == False:
		_checkTableName(db_cursor = db_cursor, table_name = table_name, exists_flag = False)
	
	# Verify any additional inputs
	# Column names
	assert type(column_names) == list, "addTable: Provided value for 'column_names' must be a list object"
	assert len(column_names) > 0, "addTable: Provided value for 'column_names' must be a non-empty list"
	assert len(set(column_names)) == len(column_names), "addTable: Provided value for 'column_names' must be a list of distinct entries"
	assert isListWithStringEntries(column_names, allow_empty_flag = False) == True, "addTable: Provided value for 'column_names' must be a list of non-empty str objects"
	# Column types
	assert type(column_types) == list, "addTable: Provided value for 'column_types' must be a list object"
	assert len(column_types) == len(column_names), "addTable: Provided value for 'column_names' must be a list of length equal to that of 'column_names'"
	assert isListWithStringEntries(column_types, allow_empty_flag = False) == True, "addTable: Provided value for 'column_types' must be a list of non-empty str objects"
	# Replace table flag
	assert type(replace_flag) == bool, "addTable: Provided value for 'replace_flag' must be a bool object"
	
	# Verify that the provided column types are valid
	for column_type in column_types:
		assert column_type in ALLOWED_COLUMN_TYPES, "addTable: Provided value for 'column_types' must contain entries only from the following list:" + str(ALLOWED_COLUMN_TYPES)
	
	# Drop the table from the db file (if needed)
	drop_table_query = "DROP TABLE IF EXISTS '" + table_name + "';"
	db_cursor.execute(drop_table_query)
	
	# Create the command for an empty version of the needed table
	create_table_query = "CREATE TABLE '" + table_name + "' (" + "\n"
	for col_index in range(len(column_names)):
		create_table_query += "\t" + column_names[col_index] + " " + column_types[col_index]
		create_table_query += ",\n" if col_index < len(column_names) - 1 else "\n);"
			
	# Execute the command
	db_cursor.execute(create_table_query)
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()
		
def deleteTable(db_path:Union[PosixPath, WindowsPath], table_name:str):
	# Delete an existing table from the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	
	# Drop the table from the database
	drop_table_query = "DROP TABLE " + table_name + ";"
	db_cursor.execute(drop_table_query)
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()
	

#########################################################################
### Define functions for reading from a db file at the given location ###
#########################################################################
def getExistingTables(db_path:Union[PosixPath, WindowsPath]) -> list:
	# Read the list of tables which currently exist in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	
	# Fetch all tables in the db file
	table_names = _checkTableName(db_cursor = db_cursor)
	
	# Close the db file connection
	db_connection.close()
	
	# Return the results
	return table_names
	
def getColumnNames(db_path:Union[PosixPath, WindowsPath], table_name:str) -> list:
	# Read the column names from the needed table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	
	# Fetch all column names from this table
	column_names = _checkColumnName(db_cursor = db_cursor, table_name = table_name)
	
	# Close the db file connection
	db_connection.close()
	
	# Return the results
	return column_names
	
def getColumnTypes(db_path:Union[PosixPath, WindowsPath], table_name:str) -> list:
	# Read the column types from the needed table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	
	# Fetch all column types from this table
	column_types_query = "SELECT type FROM PRAGMA_TABLE_INFO('" + table_name + "');"
	column_types_result = db_cursor.execute(column_types_query).fetchall()
	
	# Convert the result to the needed format
	column_types = [value[0] for value in column_types_result]
	
	# Close the db file connection
	db_connection.close()
	
	# Return the results
	return column_types
	
def getRowCount(db_path:Union[PosixPath, WindowsPath], table_name:str) -> int:
	# Read the number of rows from the needed table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	
	# Fetch the number of rows from this table
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name)
	
	# Close the db file connection
	db_connection.close()
	
	# Return the results
	return row_count

def readTable(db_path:Union[PosixPath, WindowsPath], table_name:str) -> dict:
	# Read the needed table from the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	
	# Fetch the column names and number of rows from this table
	column_names = _checkColumnName(db_cursor = db_cursor, table_name = table_name)
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name)
	
	# Read the information from the db file
	read_table_query = "SELECT * FROM '" + table_name + "';"
	read_table_result = db_cursor.execute(read_table_query).fetchall()
	
	# Convert the result to the needed format
	read_table = {}
	for col_index in range(len(column_names)):
		read_table[column_names[col_index]] = [read_table_result[row_index][col_index] for row_index in range(row_count)]
	
	# Close the db file connection
	db_connection.close()
	
	# Return the results
	return read_table
	
def readColumn(db_path:Union[PosixPath, WindowsPath], table_name:str, column_name:str) -> list:
	# Read the needed column from the given table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	_checkColumnName(db_cursor = db_cursor, table_name = table_name, column_name = column_name)
	
	# Read the information from the db file
	read_column_query = "SELECT " + column_name + " FROM '" + table_name + "';"
	read_column_result = db_cursor.execute(read_column_query).fetchall()
	
	# Convert the result to the needed format
	read_column = [value[0] for value in read_column_result]
	
	# Close the db file connection
	db_connection.close()
	
	# Return the results
	return read_column
	
def readRow(db_path:Union[PosixPath, WindowsPath], table_name:str, row_index:int) -> list:
	# Read the needed row from the given table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(row_index) == int, "readRow: Provided value for 'row_index' must be an int object"
	
	# Wrap the row index around to be in a valid range
	row_index = row_index % row_count
	
	# Read the information from the db file
	read_row_query = "SELECT * FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ";"
	read_row_result = db_cursor.execute(read_row_query).fetchone()
	
	# Convert the result to the needed format
	read_row = list(read_row_result)
		
	# Close the db file connection
	db_connection.close()
	
	# Return the results
	return read_row
	
def readEntry(db_path:Union[PosixPath, WindowsPath], table_name:str, column_name:str, row_index:int) -> Any:
	# Read the needed entry at a given row and column from the given table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	_checkColumnName(db_cursor = db_cursor, table_name = table_name, column_name = column_name)
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(row_index) == int, "readEntry: Provided value for 'row_index' must be an int object"
	
	# Wrap the row index around to be in a valid range
	row_index = row_index % row_count
	
	# Read the information from the db file
	read_entry_query = "SELECT " + column_name + " FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ";"
	read_entry_result = db_cursor.execute(read_entry_query).fetchone()
	
	# Convert the result to the needed format
	read_entry = read_entry_result[0]
	
	# Close the db file connection
	db_connection.close()
	
	# Return the results
	return read_entry
	
	
#######################################################################
### Define functions for writing to a db file at the given location ###
#######################################################################
def appendColumn(db_path:Union[PosixPath, WindowsPath], table_name:str, column_name:str, column_type:str):
	# Add a new empty column to the right of an existing table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	column_names = _checkColumnName(db_cursor = db_cursor, table_name = table_name, column_name = column_name, exists_flag = False)
	
	# Verify that the provided column types is valid
	assert column_type in ALLOWED_COLUMN_TYPES, "appendColumn: Provided value for 'column_type' be an entry from the following list:" + str(ALLOWED_COLUMN_TYPES)
	
	# Add the new column to the db file
	add_column_query = "ALTER TABLE '" + table_name + "' ADD COLUMN " + column_name + " " + column_type + ";"
	db_cursor.execute(add_column_query)
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()
	
def appendRow(db_path:Union[PosixPath, WindowsPath], table_name:str):
	# Add a new empty row to the bottom of an existing table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	column_names = _checkColumnName(db_cursor = db_cursor, table_name = table_name)
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name)
	
	# Create a list to be used as the new row for the table
	new_row = [None for _ in range(len(column_names))]
	
	# Create the query for writing the new row to the table
	write_row_query = "INSERT INTO '" + table_name + "' " + str(tuple(column_names)) + " VALUES ("
	for col_index in range(len(column_names)):
		write_row_query += "?, " if col_index < len(column_names) - 1 else "?);"
			
	# Execute the write query
	db_cursor.execute(write_row_query, new_row)
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()
	
def deleteColumn(db_path:Union[PosixPath, WindowsPath], table_name:str, column_name:str):
	# Delete a column from an existing table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	_checkColumnName(db_cursor = db_cursor, table_name = table_name, column_name = column_name)
	
	# Delete the needed column from the db file
	delete_column_query = "ALTER TABLE '" + table_name + "' DROP COLUMN " + column_name + ";"
	db_cursor.execute(delete_column_query)
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()
	
def deleteRow(db_path:Union[PosixPath, WindowsPath], table_name:str, row_index:int):
	# Delete a row from an existing table in the given db file and update the 'row_index' column
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(row_index) == int, "deleteRow: Provided value for 'row_index' must be an int object"
	
	# Wrap the row index around to be in a valid range
	row_index = row_index % row_count
	
	# Delete the needed row from the db file
	delete_row_query = "DELETE FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ";"
	db_cursor.execute(delete_row_query)
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()
	
def replaceColumn(db_path:Union[PosixPath, WindowsPath], table_name:str, column_name:str, new_column:list):
	# Replace an existing column with new values in an existing table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	_checkColumnName(db_cursor = db_cursor, table_name = table_name, column_name = column_name)
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(new_column) == list, "replaceColumn: Provided value for 'new_column' must be a list object"
	assert len(new_column) == row_count, "replaceColumn: Provided value for 'new_column' must be of length equal to the number of rows in the provided table"
	
	# Execute a sequence of replace entry queries on the entire column
	for row_index in range(row_count):
		# Create the query for replacing an existing entry in the table with a new entry (using a sub-query to allow for use of LIMIT and OFFSET)
		replace_entry_query = "UPDATE '" + table_name + "' SET " + column_name + " = ? "
		replace_entry_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ");"
				
		# Execute the write query
		db_cursor.execute(replace_entry_query, (new_column[row_index],))
	
	# Commit the changes
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()
	
def replaceRow(db_path:Union[PosixPath, WindowsPath], table_name:str, row_index:int, new_row:list):
	# Replace an existing row with new values in an existing table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	column_names = _checkColumnName(db_cursor = db_cursor, table_name = table_name)
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(row_index) == int, "replaceRow: Provided value for 'row_index' must be an int object"
	assert type(new_row) == list, "replaceRow: Provided value for 'new_row' must be a list object"
	assert len(new_row) == len(column_names), "replaceRow: Provided value for 'new_row' must be of length equal to the number of columns in the provided table"
	
	# Wrap the row index around to be in a valid range
	row_index = row_index % row_count
	
	# Create the query for replacing an existing row in the table with a new row
	# Initialize the query
	replace_row_query = "UPDATE '" + table_name + "' SET "
	# Add in the main query information
	for col_index in range(len(column_names)):
		replace_row_query += column_names[col_index] + " = ?"
		replace_row_query += ", " if col_index < len(column_names) - 1 else " "
	# Add a sub-query to allow for use of LIMIT and OFFSET
	replace_row_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ");"
			
	# Execute the write query
	db_cursor.execute(replace_row_query, new_row)
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()
	
def replaceEntry(db_path:Union[PosixPath, WindowsPath], table_name:str, column_name:str, row_index:int, new_entry:Any):
	# Replace an entry at the given row and column for an existing table in the given db file
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	column_names = _checkColumnName(db_cursor = db_cursor, table_name = table_name, column_name = column_name)
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(row_index) == int, "replaceEntry: Provided value for 'row_index' must be an int object"
	
	# Wrap the row index around to be in a valid range
	row_index = row_index % row_count

	# Create the query for replacing an existing entry in the table with a new entry (using a sub-query to allow for use of LIMIT and OFFSET)
	replace_entry_query = "UPDATE '" + table_name + "' SET " + column_name + " = ? "
	replace_entry_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ");"
			
	# Execute the write query
	db_cursor.execute(replace_entry_query, (new_entry,))
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()
	
def swapRows(db_path:Union[PosixPath, WindowsPath], table_name:str, row_index_1:int, row_index_2:int):
	# Verify the inputs which use common functions
	db_connection, db_cursor = _checkDBPath(db_path = db_path)
	_checkTableName(db_cursor = db_cursor, table_name = table_name)
	column_names = _checkColumnName(db_cursor = db_cursor, table_name = table_name)
	row_count = _checkRowCount(db_cursor = db_cursor, table_name = table_name, min_count = 2)
	
	# Verify any additional inputs
	assert type(row_index_1) == int, "swapRows: Provided value for 'row_index_1' must be an int object"
	assert type(row_index_2) == int, "swapRows: Provided value for 'row_index_2' must be an int object"
	
	# Wrap the row indices around to be in a valid range
	row_index_1 = row_index_1 % row_count
	row_index_2 = row_index_2 % row_count
	
	# Make sure that the two row indices are different
	assert row_index_1 != row_index_2, "swapRows: Provided values for 'row_index_1' and 'row_index_2' must represent different rows in the provided table"
	
	# Read the information on the 1st row from the db file
	read_row_1_query = "SELECT * FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index_1) + ";"
	read_row_1_result = db_cursor.execute(read_row_1_query).fetchone()
	read_row_1 = list(read_row_1_result)
	
	# Read the information on the 2nd row from the db file
	read_row_2_query = "SELECT * FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index_2) + ";"
	read_row_2_result = db_cursor.execute(read_row_2_query).fetchone()
	read_row_2 = list(read_row_2_result)
	
	# Write the read 2nd row to the table's 1st row
	# Initialize the query
	replace_row_1_query = "UPDATE '" + table_name + "' SET "
	# Add in the main query information
	for col_index in range(len(column_names)):
		replace_row_1_query += column_names[col_index] + " = ?"
		replace_row_1_query += ", " if col_index < len(column_names) - 1 else " "
	# Add a sub-query to allow for use of LIMIT and OFFSET
	replace_row_1_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index_1) + ");"
	# Execute the query
	db_cursor.execute(replace_row_1_query, read_row_2)
		
	# Write the read 1st row to the table's 2nd row
	# Initialize the query
	replace_row_2_query = "UPDATE '" + table_name + "' SET "
	# Add in the main query information
	for col_index in range(len(column_names)):
		replace_row_2_query += column_names[col_index] + " = ?"
		replace_row_2_query += ", " if col_index < len(column_names) - 1 else " "
	# Add a sub-query to allow for use of LIMIT and OFFSET
	replace_row_2_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index_2) + ");"
	# Execute the query
	db_cursor.execute(replace_row_2_query, read_row_1)
		
	# Commit the changes
	db_connection.commit()
	
	# Close the db file connection
	db_connection.close()