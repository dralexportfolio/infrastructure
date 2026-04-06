##########################################
### Import needed general dependencies ###
##########################################
# Built-in modules
from os.path import exists
from pathlib import Path, PosixPath, WindowsPath
from sqlite3 import connect, Connection, Cursor
from typing import Any, Tuple, Union

# Internal modules
from privacy_helper import privacyDecorator
from type_helper import isDictionaryWithStringKeys, isListWithStringEntries


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


#######################################################################################################
### Define the connection manager object to optimize commit frequency and perform additional checks ###
#######################################################################################################
# Create the decorator needed for making the attributes private
connection_manager_decorator = privacyDecorator(["_active_flag",			# class variables
												 "_buffer_size",
												 "_db_connection",
										 		 "_db_cursor",
												 "_db_path",
												 "_max_buffer_size",
												 "_checkDBPath"],			# internal functions
												 deepcopy_flag = False)		# deepcopy flag

# Define the class with private attributes
@connection_manager_decorator
class ConnectionManager:
	### Initialize the class ###
	def __init__(self, db_path:Union[PosixPath, WindowsPath], max_buffer_size:int = 100):
		# Verify the inputs
		assert type(max_buffer_size) == int, "ConnectionManager::__init__: Provided value for 'max_buffer_size' must be an int object"
		assert 1 <= max_buffer_size and max_buffer_size <= 1000, "ConnectionManager::__init__: Provided value for 'max_buffer_size' must be >= 1 and <= 1000"

		# Store the provided values
		self._db_path = db_path
		self._max_buffer_size = max_buffer_size

		# Verify the stored db path and store the associated connection and cursor
		self._checkDBPath()

		# Initialize the current buffer size
		self._buffer_size = 0

		# Initialize a flag indicating that the connection is active
		self._active_flag = True

	### Define an internal function to verify and connect to a db file using a path ###
	def _checkDBPath(self) -> Tuple[Connection, Cursor]:
		# Verify that the provided db file location is valid, returns a connection and cursor for the db file
		# Make sure the filename in the path is valid
		assert type(self._db_path) in [PosixPath, WindowsPath], "ConnectionManager::_checkDBPath: Stored value for 'db_path' must be a PosixPath or WindowsPath object"
		assert self._db_path.name.endswith(".db") == True, "ConnectionManager::_checkDBPath: Stored value for 'db_path' must refer to a filename ending with '.db'"
		assert len(self._db_path.name) > 3, "ConnectionManager::_checkDBPath: Stored value for 'db_path' must refer to a filename of length > 3"

		# Make sure the file exists
		assert exists(self._db_path.parent) == True, "ConnectionManager::_checkDBPath: Stored value for 'db_path' must refer to a filename in an existing folder"

		# Connect to the db file and create a cursor
		self._db_connection = connect(self._db_path)
		self._db_cursor = self._db_connection.cursor()

	### Define functions which are frequently used to verify database information ###
	def checkTableName(self, table_name:str = None, exists_flag:bool = True) -> list:
		# Verify that the table is (or isn't) present in the given db file (without checking existence of the db file), returns the list of table names
		# Only proceed if the connection is active
		assert self._active_flag == True, "ConnectionManager::checkTableName: Only able to commit changes to the db file when the connection is active"

		# Make sure that the table name is valid (if needed)
		if table_name is not None:
			assert type(table_name) == str, "ConnectionManager::checkTableName: If provided, value for 'table_name' must be a str object"
			assert len(table_name) > 0, "ConnectionManager::checkTableName: If provided, value for 'table_name' must be a non-empty string"

		# Fetch all tables in the database
		table_names_query = "SELECT name FROM sqlite_master WHERE type = 'table';"
		table_names_result = self._db_cursor.execute(table_names_query).fetchall()
		table_names = [value[0] for value in table_names_result]

		# Handle the various cases (if needed)
		if table_name is not None:
			if exists_flag == True:
				# Make sure the given table exists in the database
				assert table_name in table_names, "ConnectionManager::checkTableName: Provided value for 'table_name' doesn't correspond to an existing table in the db file (when it should)"
			else:
				# Make sure the given table doesn't exist in the database
				assert table_name not in table_names, "ConnectionManager::checkTableName: Provided value for 'table_name' corresponds to an existing table in the db file (when it shouldn't)"

		# Return the results
		return table_names

	def checkColumnName(self, table_name:str, column_name:str = None, exists_flag:bool = True) -> list:
		# Verify that the column name is (or isn't) present in the given table of the given db file (without checking existence of the db file or table), returns a list of column names
		# Only proceed if the connection is active
		assert self._active_flag == True, "ConnectionManager::checkColumnName: Only able to commit changes to the db file when the connection is active"

		# Make sure that the column name is valid (if needed)
		if column_name is not None:
			assert type(column_name) == str, "ConnectionManager::checkColumnName: If provided, value for 'column_name' must be a str object"
			assert len(column_name) > 0, "ConnectionManager::checkColumnName: If provided, value for 'column_name' must be a non-empty string"

		# Fetch all column names in the table
		column_names_query = "SELECT name FROM PRAGMA_TABLE_INFO('" + table_name + "');"
		column_names_result = self._db_cursor.execute(column_names_query).fetchall()
		column_names = [value[0] for value in column_names_result]

		# Handle the various cases (if needed)
		if column_name is not None:
			if exists_flag == True:
				# Make sure the given column exists in the table
				assert column_name in column_names, "ConnectionManager::checkColumnName: Provided value for 'column_name' doesn't correspond to an existing column in requested table (when it should)"
			else:
				# Make sure the given column doesn't exist in the table
				assert column_name not in column_names, "ConnectionManager::checkColumnName: Provided value for 'column_name' corresponds to an existing column in requested table (when it shouldn't)"

		# Return the results
		return column_names

	def checkRowCount(self, table_name:str, min_count:int = None) -> int:
		# Verify that the row count in the given table of the given db file is above the threshold (without checking existence of the db file or table), returns the number of rows
		# Only proceed if the connection is active
		assert self._active_flag == True, "ConnectionManager::checkRowCount: Only able to commit changes to the db file when the connection is active"

		# Make sure that the minimum row count is valid (if needed)
		if min_count is not None:
			assert type(min_count) == int, "ConnectionManager::checkRowCount: If provided, value for 'row_count' must be an int object"
			assert min_count >= 0, "ConnectionManager::checkRowCount: If provided, value for 'row_count' must be non-negative"

		# Fetch the number of rows from this table
		row_count_query = "SELECT COUNT(*) FROM '" + table_name + "';"
		row_count_result = self._db_cursor.execute(row_count_query).fetchone()
		row_count = row_count_result[0]

		# Check that the row count is large enough (if needed)
		if min_count is not None:
			assert row_count >= min_count, "ConnectionManager::checkRowCount: Relevant table doesn't have a row count >= the provided threshold"

		# Return the results
		return row_count

	#### Define function which returns internally stored values ###
	def getActiveFlag(self) -> bool:
		# Return the active flag
		return self._active_flag

	def getBufferSize(self) -> int:
		# Return the current buffer size
		return self._buffer_size

	def getConnection(self) -> Connection:
		# Return the db file connection
		return self._db_connection

	def getCursor(self) -> Cursor:
		# Return the db file cursor
		return self._db_cursor

	def getDBPath(self) -> Union[PosixPath, WindowsPath]:
		# Return the db file path
		return self._db_path

	def getMaxBufferSize(self) -> int:
		# Return the maximum buffer size
		return self._max_buffer_size

	### Define function for executing provided queries ###
	def execute(self, query:str, iterate_flag:bool, fill_values:list = None) -> Cursor:
		# Execute the given query and handle buffer behavior (if needed)
		# Only proceed if the connection is active
		assert self._active_flag == True, "ConnectionManager::execute: Only able to commit changes to the db file when the connection is active"

		# Verify the inputs (only loosely, leave detailed verification to the cursor)
		assert type(query) == str, "ConnectionManager::execute: Provided value for 'query' must be a str object"
		assert type(iterate_flag) == bool, "ConnectionManager::execute: Provided value for 'iterate_flag' must be a bool object"
		if fill_values is not None:
			assert type(fill_values) == list, "ConnectionManager::execute: If provided, value for 'fill_values' must be a list object"

		# Execute the query using the cursor
		if fill_values is None:
			self._db_cursor.execute(query)
		else:
			self._db_cursor.execute(query, fill_values)

		# Update the buffer size counter and commit changes (if needed)
		if iterate_flag == True:
			self._buffer_size += 1
			if self._buffer_size >= self._max_buffer_size:
				self.commit()

		# Return the cursor so that information can be fetched externally
		return self._db_cursor

	### Define function for manually commiting changes ###
	def commit(self):
		# Commit any outstanding changes in the buffer and reset the buffer size
		# Only proceed if the connection is active
		assert self._active_flag == True, "ConnectionManager::commit: Only able to commit changes to the db file when the connection is active"

		# Commit the changes and reset the counter
		if self._buffer_size > 0:
			self._db_connection.commit()
			self._buffer_size = 0

	### Define function to close the
	def close(self):
		# Close the connection to the db file after commiting any outstanding changes
		# Only proceed if the connection is active
		assert self._active_flag == True, "ConnectionManager::close: Only able to close the connection when the connection is active"

		# Commit any changes in the buffer
		self.commit()

		# Close the db file connection
		self._db_connection.close()

		# Indicate that the connection is no longer active
		self._active_flag = False


##########################################################################################
### Define functions for adding and deleting tables in a db file at the given location ###
##########################################################################################
def addTable(connection_manager:ConnectionManager, table_name:str, column_names:list, column_types:list, replace_flag:bool = True):
	# Add a table to the given db file (or replaces an existing table with an empty one if needed)
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "addTable: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "addTable: Provided value for 'connection_manager' must represent an active connection to a db file"
	
	# Verify any additional inputs
	# Table replace flag and table name
	assert type(replace_flag) == bool, "addTable: Provided value for 'replace_flag' must be a bool object"
	if replace_flag == False:
		connection_manager.checkTableName(table_name = table_name, exists_flag = False)
	# Column names
	assert type(column_names) == list, "addTable: Provided value for 'column_names' must be a list object"
	assert len(column_names) > 0, "addTable: Provided value for 'column_names' must be a non-empty list"
	assert len(set(column_names)) == len(column_names), "addTable: Provided value for 'column_names' must be a list of distinct entries"
	assert isListWithStringEntries(column_names, allow_empty_flag = False) == True, "addTable: Provided value for 'column_names' must be a list of non-empty str objects"
	# Column types
	assert type(column_types) == list, "addTable: Provided value for 'column_types' must be a list object"
	assert len(column_types) == len(column_names), "addTable: Provided value for 'column_names' must be a list of length equal to that of 'column_names'"
	assert isListWithStringEntries(column_types, allow_empty_flag = False) == True, "addTable: Provided value for 'column_types' must be a list of non-empty str objects"
	
	# Verify that the provided column types are valid
	for column_type in column_types:
		assert column_type in ALLOWED_COLUMN_TYPES, "addTable: Provided value for 'column_types' must contain entries only from the following list:" + str(ALLOWED_COLUMN_TYPES)
	
	# Drop the table from the db file using the connection manager (if needed)
	drop_table_query = "DROP TABLE IF EXISTS '" + table_name + "';"
	connection_manager.execute(query = drop_table_query, iterate_flag = True)
	
	# Create the command for an empty version of the needed table
	create_table_query = "CREATE TABLE '" + table_name + "' (" + "\n"
	for col_index in range(len(column_names)):
		create_table_query += "\t" + column_names[col_index] + " " + column_types[col_index]
		create_table_query += ",\n" if col_index < len(column_names) - 1 else "\n);"
			
	# Execute the command using the connection manager
	connection_manager.execute(query = create_table_query, iterate_flag = True)
		
def deleteTable(connection_manager:ConnectionManager, table_name:str):
	# Delete an existing table from the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "deleteTable: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "deleteTable: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Check that the provided table exists using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	
	# Drop the table from the database using the connection manager
	drop_table_query = "DROP TABLE " + table_name + ";"
	connection_manager.execute(query = drop_table_query, iterate_flag = True)
	

#########################################################################
### Define functions for reading from a db file at the given location ###
#########################################################################
def getExistingTables(connection_manager:ConnectionManager) -> list:
	# Read the list of tables which currently exist in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "getExistingTables: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "getExistingTables: Provided value for 'connection_manager' must represent an active connection to a db file"
	
	# Fetch all tables in the db file using the connection manager
	table_names = connection_manager.checkTableName()
	
	# Return the results
	return table_names
	
def getColumnNames(connection_manager:ConnectionManager, table_name:str) -> list:
	# Read the column names from the needed table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "getColumnNames: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "getColumnNames: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table is present using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	
	# Fetch all column names from this table using the connection manager
	column_names = connection_manager.checkColumnName(table_name = table_name)
	
	# Return the results
	return column_names
	
def getColumnTypes(connection_manager:ConnectionManager, table_name:str) -> list:
	# Read the column types from the needed table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "getColumnTypes: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "getColumnTypes: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table is present using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	
	# Fetch all column types from this table using the connection manager
	column_types_query = "SELECT type FROM PRAGMA_TABLE_INFO('" + table_name + "');"
	column_types_result = connection_manager.execute(query = column_types_query, iterate_flag = False).fetchall()
	
	# Convert the result to the needed format
	column_types = [value[0] for value in column_types_result]
	
	# Return the results
	return column_types
	
def getRowCount(connection_manager:ConnectionManager, table_name:str) -> int:
	# Read the number of rows from the needed table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "getRowCount: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "getRowCount: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table is present using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	
	# Fetch the number of rows from this table using the connection manager
	row_count = connection_manager.checkRowCount(table_name = table_name)
	
	# Return the results
	return row_count

def readTable(connection_manager:ConnectionManager, table_name:str) -> dict:
	# Read the needed table from the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "readTable: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "readTable: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table is present using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	
	# Fetch the column names and number of rows from this table using the connection manager
	column_names = connection_manager.checkColumnName(table_name = table_name)
	row_count = connection_manager.checkRowCount(table_name = table_name)
	
	# Read the information from the db file using the connection manager
	read_table_query = "SELECT * FROM '" + table_name + "';"
	read_table_result = connection_manager.execute(query = read_table_query, iterate_flag = False).fetchall()
	
	# Convert the result to the needed format
	read_table = {}
	for col_index in range(len(column_names)):
		read_table[column_names[col_index]] = [read_table_result[row_index][col_index] for row_index in range(row_count)]
	
	# Return the results
	return read_table
	
def readColumn(connection_manager:ConnectionManager, table_name:str, column_name:str) -> list:
	# Read the needed column from the given table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "readColumn: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "readColumn: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table and column name are present using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	connection_manager.checkColumnName(table_name = table_name, column_name = column_name)
	
	# Read the information from the db file using the connection manager
	read_column_query = "SELECT " + column_name + " FROM '" + table_name + "';"
	read_column_result = connection_manager.execute(query = read_column_query, iterate_flag = False).fetchall()
	
	# Convert the result to the needed format
	read_column = [value[0] for value in read_column_result]
	
	# Return the results
	return read_column
	
def readRow(connection_manager:ConnectionManager, table_name:str, row_index:int) -> list:
	# Read the needed row from the given table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "readRow: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "readRow: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table is present and that it is non-empty using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	row_count = connection_manager.checkRowCount(table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(row_index) == int, "readRow: Provided value for 'row_index' must be an int object"
	
	# Wrap the row index around to be in a valid range
	row_index = row_index % row_count
	
	# Read the information from the db file using the connection manager
	read_row_query = "SELECT * FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ";"
	read_row_result = connection_manager.execute(query = read_row_query, iterate_flag = False).fetchone()
	
	# Convert the result to the needed format
	read_row = list(read_row_result)
	
	# Return the results
	return read_row
	
def readEntry(connection_manager:ConnectionManager, table_name:str, column_name:str, row_index:int) -> Any:
	# Read the needed entry at a given row and column from the given table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "readEntry: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "readEntry: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table and column name are present using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	connection_manager.checkColumnName(table_name = table_name, column_name = column_name)

	# Get the row count for the table and make sure the table is non-empty using the connection manager
	row_count = connection_manager.checkRowCount(table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(row_index) == int, "readEntry: Provided value for 'row_index' must be an int object"
	
	# Wrap the row index around to be in a valid range
	row_index = row_index % row_count
	
	# Read the information from the db file using the connection manager
	read_entry_query = "SELECT " + column_name + " FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ";"
	read_entry_result = connection_manager.execute(query = read_entry_query, iterate_flag = False).fetchone()
	
	# Convert the result to the needed format
	read_entry = read_entry_result[0]
	
	# Return the results
	return read_entry
	
	
#######################################################################
### Define functions for writing to a db file at the given location ###
#######################################################################
def appendColumn(connection_manager:ConnectionManager, table_name:str, column_name:str, column_type:str, new_column:list = None):
	# Add a new empty column to the right of an existing table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "appendColumn: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "appendColumn: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Check that the needed table exists and that the needed column doesn't exist using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	connection_manager.checkColumnName(table_name = table_name, column_name = column_name, exists_flag = False)
	
	# Verify that the provided column types is valid
	assert column_type in ALLOWED_COLUMN_TYPES, "appendColumn: Provided value for 'column_type' be an entry from the following list:" + str(ALLOWED_COLUMN_TYPES)

	# Fetch the number of rows from this table using the connection manager
	row_count = connection_manager.checkRowCount(table_name = table_name)

	# Verify any additional inputs
	if new_column is not None:
		assert type(new_column) == list, "appendColumn: If provided, value for 'new_column' must be a list object"
		assert len(new_column) == row_count, "appendColumn: If provided, value for 'new_column' must be of length equal to the number of rows in the provided table"
		assert row_count > 0, "appendColumn: Only able to write provided value for 'new_column' to the table if the table is non-empty"

	# Add the new empty column to the table in the db file using the connection manager
	add_column_query = "ALTER TABLE '" + table_name + "' ADD COLUMN " + column_name + " " + column_type + ";"
	connection_manager.execute(query = add_column_query, iterate_flag = True)

	# Execute a sequence of replace entry queries on the entire column (if needed)
	if new_column is not None:
		for row_index in range(row_count):
			# Create the query for replacing an existing entry in the table with a new entry (using a sub-query to allow for use of LIMIT and OFFSET)
			replace_entry_query = "UPDATE '" + table_name + "' SET " + column_name + " = ? "
			replace_entry_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ");"

			# Execute the write query using the connection manager
			connection_manager.execute(query = replace_entry_query, fill_values = [new_column[row_index]], iterate_flag = True)
	
def appendRow(connection_manager:ConnectionManager, table_name:str, new_row:list = None):
	# Add a new empty row to the bottom of an existing table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "appendRow: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "appendRow: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Make sure that the needed table exists and fetch the column names using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	column_names = connection_manager.checkColumnName(table_name = table_name)

	# Verify any additional inputs
	if new_row is not None:
		assert type(new_row) == list, "appendRow: If provided, value for 'new_row' must be a list object"
		assert len(new_row) == len(column_names), "addedRow: If provided, value for 'new_row' must be of length equal to the number of columns in the provided table"

	# Use a list of all None values if new row was not provided
	if new_row is None:
		new_row = [None for _ in range(len(column_names))]

	# Create the query for adding a new row to the table
	write_row_query = "INSERT INTO '" + table_name + "' " + str(tuple(column_names)) + " VALUES ("
	for col_index in range(len(column_names)):
		write_row_query += "?, " if col_index < len(column_names) - 1 else "?);"
			
	# Add the new row to the table in the db file using the connection manager
	connection_manager.execute(query = write_row_query, fill_values = new_row, iterate_flag = True)
	
def deleteColumn(connection_manager:ConnectionManager, table_name:str, column_name:str):
	# Delete a column from an existing table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "deleteColumn: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "deleteColumn: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table and column name are present using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	connection_manager.checkColumnName(table_name = table_name, column_name = column_name)
	
	# Delete the needed column from the db file using the connection manager
	delete_column_query = "ALTER TABLE '" + table_name + "' DROP COLUMN " + column_name + ";"
	connection_manager.execute(query = delete_column_query, iterate_flag = True)
	
def deleteRow(connection_manager:ConnectionManager, table_name:str, row_index:int):
	# Delete a row from an existing table in the given db file and update the 'row_index' column
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "deleteRow: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "deleteRow: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table is present and that it is non-empty using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	row_count = connection_manager.checkRowCount(table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(row_index) == int, "deleteRow: Provided value for 'row_index' must be an int object"
	
	# Wrap the row index around to be in a valid range
	row_index = row_index % row_count
	
	# Delete the needed row from the db file using the connection manager
	# Initialize the query
	delete_row_query = "DELETE FROM '" + table_name + "' "
	# Add a sub-query to allow for use of LIMIT and OFFSET
	delete_row_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ");"
	# Execute the query using the connection manager
	connection_manager.execute(query = delete_row_query, iterate_flag = True)
	
def replaceColumn(connection_manager:ConnectionManager, table_name:str, column_name:str, new_column:list):
	# Replace an existing column with new values in an existing table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "replaceColumn: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "replaceColumn: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table and column name are present using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	connection_manager.checkColumnName(table_name = table_name, column_name = column_name)

	# Get the row count for the table and make sure the table is non-empty using the connection manager
	row_count = connection_manager.checkRowCount(table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(new_column) == list, "replaceColumn: Provided value for 'new_column' must be a list object"
	assert len(new_column) == row_count, "replaceColumn: Provided value for 'new_column' must be of length equal to the number of rows in the provided table"
	
	# Execute a sequence of replace entry queries on the entire column
	for row_index in range(row_count):
		# Create the query for replacing an existing entry in the table with a new entry (using a sub-query to allow for use of LIMIT and OFFSET)
		replace_entry_query = "UPDATE '" + table_name + "' SET " + column_name + " = ? "
		replace_entry_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ");"
				
		# Execute the write query using the connection manager
		connection_manager.execute(query = replace_entry_query, fill_values = [new_column[row_index]], iterate_flag = True)
	
def replaceRow(connection_manager:ConnectionManager, table_name:str, row_index:int, new_row:list):
	# Replace an existing row with new values in an existing table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "replaceRow: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "replaceRow: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table is present and fetch the column names using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	column_names = connection_manager.checkColumnName(table_name = table_name)

	# Get the row count for the table and make sure the table is non-empty using the connection manager
	row_count = connection_manager.checkRowCount(table_name = table_name, min_count = 1)
	
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
			
	# Execute the write query using the connection manager
	connection_manager.execute(query = replace_row_query, fill_values = new_row, iterate_flag = True)
	
def replaceEntry(connection_manager:ConnectionManager, table_name:str, column_name:str, row_index:int, new_entry:Any):
	# Replace an entry at the given row and column for an existing table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "replaceEntry: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "replaceEntry: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Check that the needed table exists, the needed column also exists, and get the existing column names using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	column_names = connection_manager.checkColumnName(table_name = table_name, column_name = column_name)

	# Get the row count for the table and make sure the table is non-empty using the connection manager
	row_count = connection_manager.checkRowCount(table_name = table_name, min_count = 1)
	
	# Verify any additional inputs
	assert type(row_index) == int, "replaceEntry: Provided value for 'row_index' must be an int object"
	
	# Wrap the row index around to be in a valid range
	row_index = row_index % row_count

	# Create the query for replacing an existing entry in the table with a new entry (using a sub-query to allow for use of LIMIT and OFFSET)
	replace_entry_query = "UPDATE '" + table_name + "' SET " + column_name + " = ? "
	replace_entry_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index) + ");"
			
	# Execute the write query using the connection manager
	connection_manager.execute(query = replace_entry_query, fill_values = [new_entry], iterate_flag= True)
	
def swapRows(connection_manager:ConnectionManager, table_name:str, row_index_1:int, row_index_2:int):
	# Swap two existing rows with each other's values in an existing table in the given db file
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "swapRows: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "swapRows: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table is present and fetch the column names using the connection manager
	connection_manager.checkTableName(table_name = table_name)
	column_names = connection_manager.checkColumnName(table_name = table_name)

	# Get the row count for the table and make sure the table has at least 2 rows using the connection manager
	row_count = connection_manager.checkRowCount(table_name = table_name, min_count = 2)
	
	# Verify any additional inputs
	assert type(row_index_1) == int, "swapRows: Provided value for 'row_index_1' must be an int object"
	assert type(row_index_2) == int, "swapRows: Provided value for 'row_index_2' must be an int object"
	
	# Wrap the row indices around to be in a valid range
	row_index_1 = row_index_1 % row_count
	row_index_2 = row_index_2 % row_count
	
	# Make sure that the two row indices are different
	assert row_index_1 != row_index_2, "swapRows: Provided values for 'row_index_1' and 'row_index_2' must represent different rows in the provided table"
	
	# Read the information on the 1st row from the db file using the connection manager
	read_row_1_query = "SELECT * FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index_1) + ";"
	read_row_1_result = connection_manager.execute(query = read_row_1_query, iterate_flag = False).fetchone()
	read_row_1 = list(read_row_1_result)
	
	# Read the information on the 2nd row from the db file using the connection manager
	read_row_2_query = "SELECT * FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index_2) + ";"
	read_row_2_result = connection_manager.execute(query = read_row_2_query, iterate_flag = False).fetchone()
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
	# Execute the query using the connection manager
	connection_manager.execute(query = replace_row_1_query, fill_values = read_row_2, iterate_flag = True)
		
	# Write the read 1st row to the table's 2nd row
	# Initialize the query
	replace_row_2_query = "UPDATE '" + table_name + "' SET "
	# Add in the main query information
	for col_index in range(len(column_names)):
		replace_row_2_query += column_names[col_index] + " = ?"
		replace_row_2_query += ", " if col_index < len(column_names) - 1 else " "
	# Add a sub-query to allow for use of LIMIT and OFFSET
	replace_row_2_query += "WHERE rowid IN (SELECT rowid FROM '" + table_name + "' LIMIT 1 OFFSET " + str(row_index_2) + ");"
	# Execute the query using the connection manager
	connection_manager.execute(query = replace_row_2_query, fill_values = read_row_1, iterate_flag = True)


##############################################################################################################
### Define a function for sorting a table in a db file in ascending or descending order for a given column ###
##############################################################################################################
def sortTable(connection_manager:ConnectionManager, table_name:str, column_name:str, ascending_flag:bool):
	# Replace the given table with a version sorted in the needed order
	# Verify the provided connection manager input
	assert type(connection_manager) == ConnectionManager, "sortTable: Provided value for 'connection_manager' must be a ConnectionManager object"
	assert connection_manager.getActiveFlag() == True, "sortTable: Provided value for 'connection_manager' must represent an active connection to a db file"

	# Verify that the needed table/column is present and fetch the table/column names using the connection manager
	table_names = connection_manager.checkTableName(table_name = table_name)
	column_names = connection_manager.checkColumnName(table_name = table_name, column_name = column_name)

	# Verify any additional inputs
	assert type(ascending_flag) == bool, "sortTable: Provided value for 'ascending_flag' must be a bool object"

	# Create the temporary table name for newly created sorted table
	temp_table_name = table_name + "_sorted"
	while temp_table_name in table_names:
		temp_table_name += "0"

	# Fetch the column types used for the original table
	column_types = getColumnTypes(connection_manager = connection_manager, table_name = table_name)

	# Create a new table with the same column names and types as the unsorted table
	# Create the command for an empty version of the needed table
	create_table_query = "CREATE TABLE '" + temp_table_name + "' (" + "\n"
	for col_index in range(len(column_names)):
		create_table_query += "\t" + column_names[col_index] + " " + column_types[col_index]
		create_table_query += ",\n" if col_index < len(column_names) - 1 else "\n);"
	# Execute the command using the connection manager
	connection_manager.execute(query = create_table_query, iterate_flag = True)

	# Insert a sorted version of the old table into the new table
	# Create the query for creating the sorted table
	sort_table_query = "INSERT INTO '" + temp_table_name + "' SELECT * FROM '" + table_name + "' ORDER BY " + column_name
	sort_table_query += " ASC;" if ascending_flag == True else " DESC;"
	# Execute the query using the connection manager
	connection_manager.execute(query = sort_table_query, iterate_flag = True)

	# Drop the old table using the connection manager
	drop_table_query = "DROP TABLE '" + table_name + "';"
	connection_manager.execute(query = drop_table_query, iterate_flag = True)

	# Rename the new table to the name of the old table using the connection manager
	rename_table_query = "ALTER TABLE '" + temp_table_name + "' RENAME TO '" + table_name + "';"
	connection_manager.execute(query = rename_table_query, iterate_flag = True)