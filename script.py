import enum
import os
import hashlib
import stat
import struct
import collections
import difflib
import zlib
import time
import operator

import urllib
from urllib import request

import requests

# Constants and Data Structures
IndexEntry = collections.namedtuple('IndexEntry', [
    'ctime_s', 'ctime_n', 'mtime_s', 'mtime_n', 'dev', 'ino', 'mode',
    'uid', 'gid', 'size', 'sha1', 'flags', 'path',
])

class GitObject:
    """Base class for Git objects"""
    def __init__(self, data=None):
        self.data = data

    def serialize(self):
        """Serialize the object data"""
        raise NotImplementedError

    def deserialize(self, data):
        """Deserialize the object data"""
        raise NotImplementedError

class Blob(GitObject):
    def serialize(self):
        return self.data

    def deserialize(self, data):
        self.data = data

class Tree(GitObject):
    def serialize(self):
        entries = []
        for mode, path, sha1 in self.data:
            # Convert mode to string and ensure it's padded to 6 digits
            mode_str = format(int(mode), '06o')  # Convert to octal string with padding
            entry = f"{mode_str} {path}\0".encode() + bytes.fromhex(sha1)
            entries.append(entry)
        return b''.join(entries)

    def deserialize(self, data):
        self.data = []
        i = 0
        while i < len(data):
            mode_end = data.index(b' ', i)
            path_end = data.index(b'\0', mode_end)
            mode = int(data[i:mode_end].decode(), 8)
            path = data[mode_end + 1:path_end].decode()
            sha1 = data[path_end + 1:path_end + 21].hex()
            self.data.append((mode, path, sha1))
            i = path_end + 21

class Commit(GitObject):
    def serialize(self):
        return self.data.encode()

    def deserialize(self, data):
        self.data = data.decode()

class ObjectType(enum.Enum):
    commit = 1
    tree = 2
    blob = 3

class GitRemoteError(Exception):
    """Custom exception for remote operations"""
    pass

def write_file(path, data, mode='wb'):
    """Write data to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode) as f:
        f.write(data)

def read_file(path):
    """Read data from a file."""
    with open(path, 'rb') as f:
        return f.read()

def init(repo):
    """Initialize a new Git repository."""
    os.makedirs(repo, exist_ok=True)
    for name in ['objects', 'refs', 'refs/heads']:
        os.makedirs(os.path.join(repo, '.git', name), exist_ok=True)
    write_file(os.path.join(repo, '.git', 'HEAD'), b'ref: refs/heads/master')
    print(f"Initialized empty repository: {repo}")

def hash_object(data, obj_type, write=True):
    """
    Hash and optionally write a git object.
    Returns: SHA-1 hash of the object.
    """
    # Prepare header and full content
    header = f"{obj_type} {len(data)}".encode()
    full_data = header + b'\x00' + data
    
    # Compute hash
    sha1 = hashlib.sha1(full_data).hexdigest()
    
    if write:
        path = os.path.join('.git', 'objects', sha1[:2], sha1[2:])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_file(path, zlib.compress(full_data))
    
    return sha1

def read_object(sha1_prefix):
    """Read a git object and return its type and data."""
    path = find_object(sha1_prefix)
    full_data = zlib.decompress(read_file(path))
    
    # Parse object header
    null_index = full_data.index(b'\x00')
    header = full_data[:null_index].decode()
    obj_type, size = header.split()
    size = int(size)
    
    # Verify size
    data = full_data[null_index + 1:]
    if len(data) != size:
        raise ValueError(f"Expected size {size}, got {len(data)} bytes")
    
    return obj_type, data

# def find_object(sha1_prefix):
#     """Find object file path by SHA-1 prefix."""
#     if len(sha1_prefix) < 2:
#         raise ValueError("SHA-1 prefix too short")
    
#     obj_dir = os.path.join('.git', 'objects', sha1_prefix[:2])
#     rest = sha1_prefix[2:]
    
#     # Find matching objects
#     try:
#         matches = [n for n in os.listdir(obj_dir) if n.startswith(rest)]
#     except FileNotFoundError:
#         matches = []
    
#     if not matches:
#         raise ValueError(f"Object {sha1_prefix} not found")
#     if len(matches) > 1:
#         raise ValueError(f"Multiple objects with prefix {sha1_prefix}")
    
#     return os.path.join(obj_dir, matches[0])

def find_object(sha1_prefix):
    if len(sha1_prefix) < 2:
        raise ValueError("SHA-1 prefix too short")

    obj_dir = os.path.join('.git', 'objects', sha1_prefix[:2])
    rest = sha1_prefix[2:]

    try:
        matches = [n for n in os.listdir(obj_dir) if n.startswith(rest)]
    except FileNotFoundError:
        raise ValueError(f"Object directory {obj_dir} not found for prefix {sha1_prefix}")

    if not matches:
        raise ValueError(f"Object {sha1_prefix} not found")
    if len(matches) > 1:
        raise ValueError(f"Multiple objects with prefix {sha1_prefix}")

    return os.path.join(obj_dir, matches[0])


# def read_index():
#     """Read the Git index file."""
#     try:
#         data = read_file(os.path.join('.git', 'index'))
#     except FileNotFoundError:
#         return []
    
#     # Verify checksum
#     digest = hashlib.sha1(data[:-20]).digest()
#     if digest != data[-20:]:
#         raise ValueError("Invalid index checksum")
    
#     # Read header
#     signature, version, num_entries = struct.unpack('!4sLL', data[:12])
#     if signature != b'DIRC':
#         raise ValueError(f"Invalid index signature {signature}")
#     if version != 2:
#         raise ValueError(f"Unknown index version {version}")
    
#     # Parse entries
#     entries = []
#     pos = 12
#     for _ in range(num_entries):
#         # Read fixed-size fields
#         fields_end = pos + 62
#         fields = struct.unpack('!LLLLLLLLLL20sH', data[pos:fields_end])
        
#         # Read path and skip padding
#         path_end = data.index(b'\0', fields_end)
#         path = data[fields_end:path_end]
#         entry = IndexEntry(*(fields + (path.decode(),)))
#         entries.append(entry)
        
#         # Move to next entry (8-byte aligned)
#         pos = path_end + 1
#         pos = (pos + 8) & ~7
    
#     return entries

def read_index():
    """Read the Git index file."""
    try:
        data = read_file(os.path.join('.git', 'index'))
    except FileNotFoundError:
        return []
    
    digest = hashlib.sha1(data[:-20]).digest()
    if digest != data[-20:]:
        raise ValueError("Invalid index checksum")
    
    signature, version, num_entries = struct.unpack('!4sLL', data[:12])
    if signature != b'DIRC':
        raise ValueError(f"Invalid index signature {signature}")
    if version != 2:
        raise ValueError(f"Unknown index version {version}")
    
    entries = []
    pos = 12
    for _ in range(num_entries):
        fields_end = pos + 62
        fields = struct.unpack('!LLLLLLLLLL20sH', data[pos:fields_end])
        
        path_end = data.index(b'\0', fields_end)
        path = data[fields_end:path_end]
        
        # Try different encodings, fallback to raw bytes if needed
        try:
            path_str = path.decode('utf-8')
        except UnicodeDecodeError:
            try:
                path_str = path.decode('latin-1')
            except UnicodeDecodeError:
                path_str = str(path)[2:-1]  # Convert bytes to string representation
        
        entry = IndexEntry(*(fields + (path_str,)))
        entries.append(entry)
        
        pos = path_end + 1
        pos = (pos + 8) & ~7
    
    return entries

def write_index(entries):
    """Write the Git index file."""
    # Prepare index data
    chunks = []
    
    # Write header
    chunks.append(struct.pack('!4sLL', b'DIRC', 2, len(entries)))
    
    # Write entries
    for entry in entries:
        chunks.append(struct.pack('!LLLLLLLLLL20sH',
            entry.ctime_s, entry.ctime_n,
            entry.mtime_s, entry.mtime_n,
            entry.dev, entry.ino, entry.mode,
            entry.uid, entry.gid, entry.size,
            entry.sha1, entry.flags))
        chunks.append(entry.path.encode() + b'\0')
        
        # Add padding to maintain 8-byte alignment
        pad = 8 - ((62 + len(entry.path) + 1) % 8)
        if pad < 8:
            chunks.append(b'\0' * pad)
    
    # Concatenate chunks and add checksum
    data = b''.join(chunks)
    chunks.append(hashlib.sha1(data).digest())
    
    # Write to file
    write_file(os.path.join('.git', 'index'), b''.join(chunks))

def add(paths):
    """Add files to the Git index."""
    paths = [p.replace('\\', '/') for p in paths]
    
    # Read current index
    all_entries = read_index()
    entries = [e for e in all_entries if e.path not in paths]
    
    # Add new entries
    for path in paths:
        # Read and hash file
        data = read_file(path)
        sha1 = hash_object(data, 'blob')
        
        # Create new index entry
        st = os.stat(path)
        flags = len(path.encode())
        
        entry = IndexEntry(
            int(st.st_ctime), 0,
            int(st.st_mtime), 0,
            st.st_dev, st.st_ino,
            st.st_mode, st.st_uid, st.st_gid,
            st.st_size,
            bytes.fromhex(sha1), flags,
            path)
        entries.append(entry)
    
    # Sort and write index
    entries.sort(key=operator.attrgetter('path'))
    write_index(entries)

def write_tree():
    """Create a tree object from the current index."""
    entries = []
    for entry in read_index():
        mode = "{:o}".format(entry.mode)
        entries.append((mode, entry.path, entry.sha1.hex()))
    
    # Create tree object
    tree = Tree(entries)
    tree_data = tree.serialize()
    return hash_object(tree_data, 'tree')

def commit(message, author=None):
    """Create a commit object."""
    if author is None:
        author = f"{os.environ.get('GIT_AUTHOR_NAME', 'Unknown')} <{os.environ.get('GIT_AUTHOR_EMAIL', 'unknown@example.com')}>"
    
    # Get tree hash
    tree = write_tree()
    
    # Get parent hash
    parent = None
    head_path = os.path.join('.git', 'refs', 'heads', 'master')
    if os.path.exists(head_path):
        parent = read_file(head_path, 'r').strip()
    
    # Build commit object
    timestamp = int(time.time())
    timezone = time.strftime("%z")
    commit_content = []
    commit_content.append(f"tree {tree}")
    if parent:
        commit_content.append(f"parent {parent}")
    commit_content.extend([
        f"author {author} {timestamp} {timezone}",
        f"committer {author} {timestamp} {timezone}",
        "",
        message,
        ""
    ])
    
    # Create and write commit object
    commit_data = "\n".join(commit_content).encode()
    sha1 = hash_object(commit_data, 'commit')
    
    # Update master ref
    write_file(head_path, f"{sha1}\n".encode())
    print(f"[master {sha1[:7]}] {message}")
    return sha1

def get_status():
    """Get status of working directory."""
    entries = {e.path: e for e in read_index()}
    paths = set()
    
    # Scan working directory
    for root, _, files in os.walk('.'):
        if root.startswith('./.git'):
            continue
        for file in files:
            path = os.path.join(root, file)[2:].replace('\\', '/')
            paths.add(path)
    
    # Compare working directory with index
    changed = set()
    for path in paths & set(entries):
        data = read_file(path)
        sha1 = hash_object(data, 'blob', write=False)
        if sha1 != entries[path].sha1.hex():
            changed.add(path)
    
    # Find new and deleted files
    new = paths - set(entries)
    deleted = set(entries) - paths
    
    return sorted(changed), sorted(new), sorted(deleted)

def status():
    """Show status of working directory."""
    changed, new, deleted = get_status()
    
    if changed:
        print("Changes not staged for commit:")
        for path in changed:
            print(f"    modified: {path}")
    
    if new:
        print("Untracked files:")
        for path in new:
            print(f"    {path}")
            
    if deleted:
        print("Deleted files:")
        for path in deleted:
            print(f"    {path}")

def diff():
    """Show changes between index and working directory."""
    changed, _, _ = get_status()
    
    for path in changed:
        # Read working copy
        try:
            with open(path, 'r', encoding='utf-8') as f:
                working_data = f.read().splitlines()
        except UnicodeDecodeError:
            print(f"Binary file {path} differs")
            continue
        
        # Read index version
        entry = next(e for e in read_index() if e.path == path)
        obj_type, index_data = read_object(entry.sha1.hex())
        try:
            # Convert bytes to string and split into lines
            index_data = index_data.decode('utf-8').splitlines()
        except UnicodeDecodeError:
            print(f"Binary file {path} differs")
            continue
        
        # Generate diff
        diff_lines = difflib.unified_diff(
            index_data, working_data,
            f'a/{path}', f'b/{path}',
            lineterm='')
        
        print(f"diff --git a/{path} b/{path}")
        print('\n'.join(list(diff_lines)))

def get_local_master_hash():
    """Get current commit hash (SHA-1 string) of local master branch."""
    master_path = os.path.join('.git', 'refs', 'heads', 'master')
    try:
        return read_file(master_path).decode().strip()
    except FileNotFoundError:
        return None
    
def extract_lines(data):
    """Extract list of lines from given server data."""
    lines = []
    i = 0
    for _ in range(1000):
        line_length = int(data[i:i + 4], 16)
        line = data[i + 4:i + line_length]
        lines.append(line)
        if line_length == 0:
            i += 4
        else:
            i += line_length
        if i >= len(data):
            break
    return lines


def build_lines_data(lines):
    """Build byte string from given lines to send to server."""
    result = []
    for line in lines:
        result.append('{:04x}'.format(len(line) + 5).encode())
        result.append(line)
        result.append(b'\n')
    result.append(b'0000')
    return b''.join(result)


# def http_request(url, username, password):
#     response = requests.get(url, auth=(username, password))
#     response.raise_for_status()
#     return response.content

def http_request(url, username, password, data=None):
    """Make HTTP request to given URL with authentication and optional data."""
    headers = {'Content-Type': 'application/x-git-upload-pack-request'} if data else {}
    method = 'POST' if data else 'GET'
    
    response = requests.request(
        method=method,
        url=url,
        auth=(username, password),
        data=data,
        headers=headers
    )
    response.raise_for_status()
    return response.content


def get_remote_master_hash(git_url, username, password):
    """Get commit hash of remote master branch, return SHA-1 hex string or
    None if no remote commits.
    """
    url = git_url + '/info/refs?service=git-receive-pack'
    response = http_request(url, username, password)
    lines = extract_lines(response)
    assert lines[0] == b'# service=git-receive-pack\n'
    assert lines[1] == b''
    if lines[2][:40] == b'0' * 40:
        return None
    master_sha1, master_ref = lines[2].split(b'\x00')[0].split()
    assert master_ref == b'refs/heads/master'
    assert len(master_sha1) == 40
    return master_sha1.decode()

def read_tree(sha1=None, data=None):
    """Read tree object with given SHA-1 (hex string) or data, and return list
    of (mode, path, sha1) tuples.
    """
    if sha1 is not None:
        obj_type, data = read_object(sha1)
        assert obj_type == 'tree'
    elif data is None:
        raise TypeError('must specify "sha1" or "data"')
    
    i = 0
    entries = []
    while i < len(data):
        end = data.find(b'\x00', i)
        if end == -1:
            break
            
        entry = data[i:end].decode()
        space_pos = entry.find(' ')
        if space_pos == -1:
            raise ValueError(f"Invalid tree entry format: {entry}")
            
        mode_str = entry[:space_pos].strip()
        path = entry[space_pos + 1:]
        
        try:
            # Git uses specific mode values
            mode = int(mode_str, 8) & 0o777777
        except ValueError:
            raise ValueError(f"Invalid mode format in tree entry: {mode_str}")
            
        digest = data[end + 1:end + 21]
        entries.append((mode, path, digest.hex()))
        i = end + 1 + 20
    
    return entries

def find_tree_objects(tree_sha1):
    """Return set of SHA-1 hashes of all objects in this tree (recursively),
    including the hash of the tree itself.
    """
    objects = {tree_sha1}
    for mode, path, sha1 in read_tree(sha1=tree_sha1):
        # Git uses 040000 for directories
        if (mode & 0o170000) == 0o040000:
            objects.update(find_tree_objects(sha1))
        else:
            objects.add(sha1)
    return objects

def find_commit_objects(commit_sha1):
    """Return set of SHA-1 hashes of all objects in this commit (recursively),
    its tree, its parents, and the hash of the commit itself.
    """
    objects = {commit_sha1}
    obj_type, commit = read_object(commit_sha1)
    assert obj_type == 'commit'
    lines = commit.decode().splitlines()
    tree = next(l[5:45] for l in lines if l.startswith('tree '))
    objects.update(find_tree_objects(tree))
    parents = (l[7:47] for l in lines if l.startswith('parent '))
    for parent in parents:
        objects.update(find_commit_objects(parent))
    return objects


def find_missing_objects(local_sha1, remote_sha1):
    """Return set of SHA-1 hashes of objects in local commit that are missing
    at the remote (based on the given remote commit hash).
    """
    local_objects = find_commit_objects(local_sha1)
    if remote_sha1 is None:
        return local_objects
    remote_objects = find_commit_objects(remote_sha1)
    return local_objects - remote_objects


def encode_pack_object(obj):
    """Encode a single object for a pack file and return bytes (variable-
    length header followed by compressed data bytes).
    """
    obj_type, data = read_object(obj)
    type_num = ObjectType[obj_type].value
    size = len(data)
    byte = (type_num << 4) | (size & 0x0f)
    size >>= 4
    header = []
    while size:
        header.append(byte | 0x80)
        byte = size & 0x7f
        size >>= 7
    header.append(byte)
    return bytes(header) + zlib.compress(data)


def create_pack(objects):
    """Create pack file containing all objects in given given set of SHA-1
    hashes, return data bytes of full pack file.
    """
    header = struct.pack('!4sLL', b'PACK', 2, len(objects))
    body = b''.join(encode_pack_object(o) for o in sorted(objects))
    contents = header + body
    sha1 = hashlib.sha1(contents).digest()
    data = contents + sha1
    return data


def push(git_url, username=None, password=None):
    """Push master branch to given git repo URL."""
    if username is None:
        username = os.environ['GIT_USERNAME']
    if password is None:
        password = os.environ['GIT_PASSWORD']
    remote_sha1 = get_remote_master_hash(git_url, username, password)
    local_sha1 = get_local_master_hash()
    missing = find_missing_objects(local_sha1, remote_sha1)
    print('updating remote master from {} to {} ({} object{})'.format(
            remote_sha1 or 'no commits', local_sha1, len(missing),
            '' if len(missing) == 1 else 's'))
    lines = ['{} {} refs/heads/master\x00 report-status'.format(
            remote_sha1 or ('0' * 40), local_sha1).encode()]
    data = build_lines_data(lines) + create_pack(missing)
    url = git_url + '/git-receive-pack'
    response = http_request(url, username, password, data=data)
    lines = extract_lines(response)
    assert len(lines) >= 2, \
        'expected at least 2 lines, got {}'.format(len(lines))
    assert lines[0] == b'unpack ok\n', \
        "expected line 1 b'unpack ok', got: {}".format(lines[0])
    assert lines[1] == b'ok refs/heads/master\n', \
        "expected line 2 b'ok refs/heads/master\n', got: {}".format(lines[1])
    return (remote_sha1, missing)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Git implementation")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # init
    init_parser = subparsers.add_parser('init', help='Initialize a new repository')
    init_parser.add_argument('path', help='Where to create the repository')
    
    # add
    add_parser = subparsers.add_parser('add', help='Add files to the index')
    add_parser.add_argument('paths', nargs='+', help='Paths of files to add')
    
    # commit
    commit_parser = subparsers.add_parser('commit', help='Record changes to the repository')
    commit_parser.add_argument('-m', '--message', required=True, help='Commit message')
    commit_parser.add_argument('-a', '--author', help='Author of the commit')
    
    # status
    subparsers.add_parser('status', help='Show working tree status')
    
    # diff
    subparsers.add_parser('diff', help='Show changes between index and working tree')

    # push
    push_parser = subparsers.add_parser('push', help='Push to remote repository')
    push_parser.add_argument('url', help='Remote repository URL')
    push_parser.add_argument('--username', required=True, help='Username')
    push_parser.add_argument('--password', required=True, help='Password')
    push_parser.add_argument('--branch', default='master', help='Branch to push')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        init(args.path)
    elif args.command == 'add':
        add(args.paths)
    elif args.command == 'commit':
        commit(args.message, args.author)
    elif args.command == 'status':
        status()
    elif args.command == 'diff':
        diff()
    elif args.command == 'push':
        push(args.url, args.username, args.password)

if __name__ == '__main__':
    main()