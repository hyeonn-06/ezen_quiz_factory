-- sql plus / as sysdba
-- alter session set container=EMPPDB
-- show con_name   

create pluggable database "QFPDB" admin user "QFADMIN" identified by "1111" roles=(dba) file_name_convert = ('pdbseed','QFPDB') ;

alter pluggable database "QFPDB" open read write; 
alter pluggable database "QFPDB" save state ;

alter session set container = "QFPDB";  
grant resource, dba to "QFADMIN"; 

conn QFADMIN/1111@localhost/QFPDB 
CREATE tablespace qftbls 
datafile 'qftbls' size 64M autoextend on; 

CREATE USER qf IDENTIFIED BY 1111 default tablespace qftbls ;
alter user qf quota unlimited on qftbls ;
GRANT connect, resource, dba TO qf ;

connect qf/1111@localhost/qfpdb

commit;






