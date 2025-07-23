#include <bits/stdc++.h>
using namespace std;

struct Node {
    int par = -1;
    vector<int> ch;
    bool locked = false;
    int uid = -1;
    int sub = 0;
};

class Tree {
private:
    vector<Node> g;
    unordered_map<string, int> id;
    int k;

    bool hasLockedAnc(int u) const {
        for (int p = g[u].par; p != -1; p = g[p].par)
            if (g[p].locked) return true;
        return false;
    }
    void updateAnc(int u, int d) {
        for (int p = u; p != -1; p = g[p].par) g[p].sub += d;
    }
    bool collectLocks(int u, int uid, vector<int>& lst) {
        if (!g[u].sub) return true;
        for (int v : g[u].ch) {
            if (!g[v].sub) continue;
            if (g[v].locked) {
                if (g[v].uid != uid) return false;
                lst.push_back(v);
            }
            if (!collectLocks(v, uid, lst)) return false;
        }
        return true;
    }
public:
    Tree(int n, int m, const vector<string>& nm) : g(n), id(), k(m) {
        for (int i = 0; i < n; ++i) {
            id[nm[i]] = i;
            if (i) {
                int p = (i - 1) / k;
                g[i].par = p;
                g[p].ch.push_back(i);
            }
        }
    }
    int idx(const string& name) const { return id.at(name); }
    bool lock(int u, int uid) {
        Node& n = g[u];
        if (n.locked || n.sub || hasLockedAnc(u)) return false;
        n.locked = true;
        n.uid = uid;
        updateAnc(u, 1);
        return true;
    }
    bool unlock(int u, int uid) {
        Node& n = g[u];
        if (!n.locked || n.uid != uid) return false;
        n.locked = false;
        n.uid = -1;
        updateAnc(u, -1);
        return true;
    }
    bool upgrade(int u, int uid) {
        Node& n = g[u];
        if (n.locked || hasLockedAnc(u) || !n.sub) return false;
        vector<int> lst;
        if (!collectLocks(u, uid, lst)) return false;
        for (int v : lst) unlock(v, uid);
        return lock(u, uid);
    }
};

class LockManager {
private:
    Tree& tr;
public:
    explicit LockManager(Tree& t) : tr(t) {}
    bool exec(int i, int uid) { return tr.lock(i, uid); }
};

class UnlockManager {
private:
    Tree& tr;
public:
    explicit UnlockManager(Tree& t) : tr(t) {}
    bool exec(int i, int uid) { return tr.unlock(i, uid); }
};

class UpgradeManager {
private:
    Tree& tr;
public:
    explicit UpgradeManager(Tree& t) : tr(t) {}
    bool exec(int i, int uid) { return tr.upgrade(i, uid); }
};

class QueryManager {
public:
    void run() {
        ios::sync_with_stdio(false);
        cin.tie(nullptr);
        int n, m, q;
        if (!(cin >> n)) return;
        cin >> m >> q;
        vector<string> names(n);
        for (string& s : names) cin >> s;
        Tree tree(n, m, names);
        LockManager    lckMgr(tree);
        UnlockManager  unlMgr(tree);
        UpgradeManager upgMgr(tree);
        while (q--) {
            int op, uid;
            string name;
            cin >> op >> name >> uid;
            bool ok = (op == 1) ? lckMgr.exec(tree.idx(name), uid)
                     : (op == 2) ? unlMgr.exec(tree.idx(name), uid)
                     : upgMgr.exec(tree.idx(name), uid);
            cout << (ok ? "true" : "false") << '\n';
        }
    }
};

int main() {
    QueryManager().run();
    return 0;
}